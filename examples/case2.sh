export EXTERNAL_MEMORY_TEXT="But to open the session I'm very pleased that Christian Bason the director of the Danish Design center in Copenhagen will lead us into his view of policy innovation I believe there is no other person in Europe. Maybe on the planet who has more experience in policy innovation than Christian Bason"
python -m audio_agent.examples.demo_run_api_asr \
    --audio /data/test_oracle_v1/data/format.1/data_wav.ark:16920526 \
    --image /data/test_oracle_v1/slides/child_0000/child_0000-00004.png \
    --question 'Transcribe what is being said and correct domain terms using the image and history. Return only valid JSON with exactly this schema: {"transcription": "<corrected transcript text>"}' \
    --frontend-model "qwen3-omni-flash" \
    --planner-model "qwen2.5-7b-instruct" \
    --max-tokens 2048