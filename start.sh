source /workspace/sh/env

UV_PROJECT_ENVIRONMENT=/root/tmp/env

NDIF_API_KEY=adbcde9f-bc87-4d14-8d47-a39a334db8c0
HF_TOKEN=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

export NDIF_API_KEY
export HF_TOKEN
export UV_PROJECT_ENVIRONMENT

# add the following public key to ssh directory, start the ssh agent and add the key

cp /workspace/.ssh/config /root/.ssh/
cp /workspace/.ssh/id_ed25519 /root/.ssh/
cp /workspace/.ssh/id_ed25519.pub /root/.ssh/
chmod 600 /root/.ssh/id_ed25519
chmod 600 /root/.ssh/id_ed25519.pub
chmod 700 /root/.ssh
eval "$(ssh-agent -s)"
ssh-add /root/.ssh/id_ed25519

cd /workspace/sandbox && uv sync 