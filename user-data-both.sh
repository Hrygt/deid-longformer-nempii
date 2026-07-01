#!/bin/bash
set -ex

# Log everything
exec > >(tee /var/log/user-data.log) 2>&1
echo "=== CPT DNN + Clinical De-id Service Setup ==="

# Update system
yum update -y

# ============================================
# CPT DNN SERVICE
# ============================================
echo "=== Setting up CPT DNN Service ==="

mkdir -p /opt/cpt-dnn
cd /opt/cpt-dnn

aws s3 sync s3://cpt-dnn-model-artifacts-675138611834/model/ /opt/cpt-dnn/model/
aws s3 sync s3://cpt-dnn-model-artifacts-675138611834/app/ /opt/cpt-dnn/app/

/opt/pytorch/bin/pip install transformers fastapi uvicorn boto3

TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
ALLOCATION_ID="eipalloc-015559bd652771d48"
aws ec2 associate-address --instance-id $INSTANCE_ID --allocation-id $ALLOCATION_ID --region us-west-2 || true

cd /opt/cpt-dnn/app
ln -sf ../model model

cat > /etc/systemd/system/cpt-dnn.service << 'SYSTEMD'
[Unit]
Description=CPT DNN FastAPI Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/cpt-dnn/app
Environment="PATH=/opt/pytorch/bin:/usr/local/bin:/usr/bin"
Environment="AWS_DEFAULT_REGION=us-west-2"
ExecStart=/opt/pytorch/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SYSTEMD

# ============================================
# CLINICAL DE-ID SERVICE
# ============================================
echo "=== Setting up Clinical De-id Service ==="

mkdir -p /opt/clinical-deid
cd /opt/clinical-deid

git clone https://github.com/Hrygt/clinical-deid.git temp
# Fix A: capture the deployed commit SHA before `mv`/`rm -rf` discards temp/.git,
# so it can be surfaced at /deid/health (VERSION is not a repo-tracked file, so the
# `mv temp/* .` below cannot clobber it).
(cd temp && git rev-parse HEAD) > /opt/clinical-deid/VERSION
mv temp/* . && rm -rf temp

/opt/pytorch/bin/pip install huggingface_hub faker
/opt/pytorch/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('riggsmed/clinical-deid', local_dir='model')"

mkdir -p /opt/clinical-deid/static
aws s3 sync s3://cpt-dnn-model-artifacts-675138611834/clinical-deid/static/ /opt/clinical-deid/static/

/opt/pytorch/bin/python << 'PYEOF'
import re
with open('/opt/clinical-deid/api.py', 'r') as f:
    content = f.read()
if 'StaticFiles' not in content:
    content = content.replace(
        'from fastapi import FastAPI',
        'from fastapi import FastAPI\nfrom fastapi.staticfiles import StaticFiles\nfrom fastapi.responses import FileResponse'
    )
if 'app.mount("/static"' not in content:
    content = re.sub(
        r'(app = FastAPI\([^)]*\))',
        r'\1\n\napp.mount("/static", StaticFiles(directory="static"), name="static")\n\n@app.get("/")\nasync def root():\n    return FileResponse("static/index.html")',
        content
    )
with open('/opt/clinical-deid/api.py', 'w') as f:
    f.write(content)
PYEOF

cat > /etc/systemd/system/clinical-deid.service << 'SYSTEMD'
[Unit]
Description=Clinical De-identification FastAPI Service
# Fix B: reconcile the box to the pinned SHA before the app starts.
Wants=clinical-deid-reconcile.service
After=network.target clinical-deid-reconcile.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/clinical-deid
Environment="PATH=/opt/pytorch/bin:/usr/local/bin:/usr/bin"
Environment="DEID_MODEL_PATH=/opt/clinical-deid/model"
ExecStart=/opt/pytorch/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SYSTEMD

# Fix B: pinned-SHA reconcile oneshot. Runs at boot/relaunch BEFORE the app, so a relaunch
# self-heals to the intended SHA. It does NOT re-run on the app's crash-restart (that restarts
# only clinical-deid.service), so a crash-loop cannot hammer S3. It runs a COPY of reconcile.sh
# from /run so the overlay can safely overwrite the on-disk script mid-run. Reconcile only ever
# moves prod to the human-set pin — never to unreviewed main.
cat > /etc/systemd/system/clinical-deid-reconcile.service << 'SYSTEMD'
[Unit]
Description=Clinical De-id deploy reconcile (pin -> /opt/clinical-deid on boot)
Wants=network-online.target
After=network-online.target
Before=clinical-deid.service

[Service]
Type=oneshot
RemainAfterExit=yes
Environment="PATH=/opt/pytorch/bin:/usr/local/bin:/usr/bin"
Environment="AWS_DEFAULT_REGION=us-west-2"
ExecStart=/bin/bash -c 'install -m0755 /opt/clinical-deid/reconcile.sh /run/clinical-deid-reconcile.sh && /run/clinical-deid-reconcile.sh'

[Install]
WantedBy=multi-user.target
SYSTEMD

# ============================================
# START ALL SERVICES
# ============================================
systemctl daemon-reload
systemctl enable cpt-dnn
systemctl start cpt-dnn
systemctl enable clinical-deid-reconcile
systemctl enable clinical-deid
systemctl start clinical-deid

echo "=== Setup Complete ==="