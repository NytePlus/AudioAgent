export EXTERNAL_MEMORY_TEXT="And also with us is so Suellen Carvalho she is the innovation director strategic designer and changemaker helping governments companies and N G Os to design solutions and impact people's lives. She has led projects teamed in women empowerment education health social and economic development. So thank you everyone for being here with me ah thank you Terry and Antonio. Um I am Suellen Carvalho I'm an innovation director at Tellus institute um next one please call it"
python -m audio_agent.examples.demo_run_api_asr \
    --audio /data/test_oracle_v1/data/format.1/data_wav.ark:53936654 \
    --image /data/test_oracle_v1/slides/design_0001/design_0001-00007.png \
    --question "Transcribe what is being said and correct domain terms using the image." \
    --frontend-model "qwen2.5-omni-7b" \
    --planner-model "qwen2.5-7b-instruct" \
    --max-tokens 2048