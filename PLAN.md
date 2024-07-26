1. Each change in code = new commit before running an experiment.
This way all experiments on the same commit share the same code and it's easier to compare them etc.
Ideally also each commit would have a different view defined (in DVC studio or extension if possible)

2. Create a "preprocessor" which generates synthetic MIDI data for training one vs all type of classifiers.
The parameters can include:
    - min_note_duration
    - max_note_duration
    - max_silmultaneous_notes
    - notes_vocab
    - training_data_size_per_note
    - volume_max
    - volume_min
The preprocessor should generate midi track(s) for each note such that ~50% of the time the note of interest is playing,
while another ~50% of the time it's not.
Then it would be possible to evaluate how well different types of models (perhaps starting from simplest ones like linear regression / SVMs / tree ensambles / k-nearest etc., then maybe trying out some NN's) are able to differentiate between the playing vs non-playing state of a note based on ~3-9 steps of
10ms-hop-length spectogram (cenetered at the current timeframe). Note that with 200 filters, 9 steps of such spectrogram would have 1800 features.

This approach makes it easier to evaluate:
- Whether given type of model is the right fit for the task
- How well-suited a spectogram with given parameters is for this task
- Whether changing min_note_duration to a higher number can improves the accuracy
- How much harder it becomes for the model to classify datasets with different values of max_silmultaneous_notes
- See how the changes in volume make the distinctions easier / harder
- **What are the common mistakes the model makes (for example: distinguishing between notes of similar frequencies, trouble spotting the correct start/end time of the note etc.)**
- What is the good size of a dataset that allows the model to achieve high accuracy on one-vs-all task. What is the size beyond which there isn't much of an improvement.

3. **Note the conclusions** from 2 (create a document like CONCLUSIONS.md and try to make it short and on-point. You can format it similar to a typical CHANGELOG)

4. Introduce different instruments and evaluate again. See whether the models still perform well when:
- Predicting whether a given note is playing (regardless of an instrument)
- Predicting whether a given note is being played ON a given instrument (while predicting False if the note is the same but the instrument differs)

5. Based on the conclusions start planning further tasks like:
- Design a reasonable pipeline for the Google Cloud integration w/ experiment tracking etc.
- Create a full training dataset which includes:
    - Real audio data from musicnet (possibly)
    - Midi data from musicnet played using different midi soundfonts + also potentially some random noise
    - Synthetic midi data which focuses on common model mistakes + different soundfonts + potentially some random noise