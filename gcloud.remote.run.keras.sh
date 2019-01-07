gcloud ml-engine jobs submit training JOB5 \
    --module-name=trainer.cnn_with_keras \
    --package-path=./trainer \
    --job-dir=gs://keras-on-cloud3 \
    --region=us-central1 \
    --config=trainer/cloudml-gpu.yaml
