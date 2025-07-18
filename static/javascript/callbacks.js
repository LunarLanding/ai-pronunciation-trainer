// Audio context initialization
let mediaRecorder, audioChunks, audioBlob, stream, audioRecorded;
const ctx = new AudioContext();
let currentAudioForPlaying;
let lettersOfWordAreCorrect = [];

// UI-related variables
const page_title = "AI Pronunciation Trainer";

const getThemeColor = (cssVar) =>
    getComputedStyle(document.documentElement).getPropertyValue(cssVar).trim();

const accuracy_colors = [
    getThemeColor('--color-good'),
    getThemeColor('--color-okay'),
    getThemeColor('--color-bad')
];
let badScoreThreshold = 30;
let mediumScoreThreshold = 70;
let currentSample = 0;
let currentScore = 0.;
let sample_difficult = 0;
let scoreMultiplier = 1;
let playAnswerSounds = true;
let isNativeSelectedForPlayback = true;
let isRecording = false;
let serverIsInitialized = false;
let serverWorking = true;
let languageFound = true;
let currentSoundRecorded = false;
let currentText, currentIpa, real_transcripts_ipa, matched_transcripts_ipa;
let wordCategories;
let startTime, endTime;

// API related variables 
let AILanguage = "de"; // Standard is German


let STScoreAPIKey = 'rll5QsTiv83nti99BW6uCmvs9BDVxSB39SVFceYb'; // Public Key. If, for some reason, you would like a private one, send-me a message and we can discuss some possibilities
let apiMainPathSample = '';// 'http://127.0.0.1:3001';// 'https://a3hj0l2j2m.execute-api.eu-central-1.amazonaws.com/Prod';
let apiMainPathSTS = '';// 'https://wrg7ayuv7i.execute-api.eu-central-1.amazonaws.com/Prod';


// Variables to playback accuracy sounds
let soundsPath = '/static/sounds';//'https://stscore-sounds-bucket.s3.eu-central-1.amazonaws.com';
let soundFileGood = null;
let soundFileOkay = null;
let soundFileBad = null;

// Speech generation
var synth = window.speechSynthesis;
let voice_idx = 0;
let voice_synth = null;

//############################ UI general control functions ###################
const unblockUI = () => {
    document.getElementById("recordAudio").classList.remove('disabled');
    document.getElementById("playSampleAudio").classList.remove('disabled');
    document.getElementById("buttonNext").onclick = () => getNextSample();
    document.getElementById("buttonNext").classList.remove('disabled');
    document.getElementById("original_script").classList.remove('disabled');
    document.getElementById("buttonNext").style["background-color"] = '#58636d';

    if (currentSoundRecorded)
        document.getElementById("playRecordedAudio").classList.remove('disabled');


};

const blockUI = () => {

    document.getElementById("recordAudio").classList.add('disabled');
    document.getElementById("playSampleAudio").classList.add('disabled');
    document.getElementById("buttonNext").onclick = null;
    document.getElementById("original_script").classList.add('disabled');
    document.getElementById("playRecordedAudio").classList.add('disabled');

    document.getElementById("buttonNext").style["background-color"] = '#adadad';


};

const UIError = (error) => {
    blockUI();
    document.getElementById("buttonNext").onclick = () => getNextSample(); //If error, user can only try to get a new sample
    document.getElementById("buttonNext").style["background-color"] = '#58636d';

    document.getElementById("recorded_ipa_script").innerHTML = "";
    document.getElementById("ipa_script").innerHTML = ""

    document.getElementById("main_title").innerHTML = 'Server Error';
    document.getElementById("original_script").innerHTML = 'Server error. Either the daily quota of the server is over or there was some internal error. You can try to generate a new sample in a few seconds. If the error persist, try comming back tomorrow or download the local version from Github :)';
    throw error;
};

const UINotSupported = () => {
    unblockUI();

    document.getElementById("main_title").innerHTML = "Browser unsupported";

}

const UIRecordingError = () => {
    unblockUI();
    document.getElementById("main_title").innerHTML = "Recording error, please restart page.";
}



//################### Application state functions #######################
function updateScore(currentPronunciationScore) {

    if (isNaN(currentPronunciationScore))
        return;
    currentScore += currentPronunciationScore * scoreMultiplier;
    currentScore = Math.round(currentScore);
}

const cacheSoundFiles = async () => {
    await fetch(soundsPath + '/ASR_good.wav').then(data => data.arrayBuffer()).
        then(arrayBuffer => ctx.decodeAudioData(arrayBuffer)).
        then(decodeAudioData => {
            soundFileGood = decodeAudioData;
        });

    await fetch(soundsPath + '/ASR_okay.wav').then(data => data.arrayBuffer()).
        then(arrayBuffer => ctx.decodeAudioData(arrayBuffer)).
        then(decodeAudioData => {
            soundFileOkay = decodeAudioData;
        });

    await fetch(soundsPath + '/ASR_bad.wav').then(data => data.arrayBuffer()).
        then(arrayBuffer => ctx.decodeAudioData(arrayBuffer)).
        then(decodeAudioData => {
            soundFileBad = decodeAudioData;
        });
}

const getNextSample = async () => {



    blockUI();

    if (!serverIsInitialized)
        await initializeServer();

    if (!serverWorking) {
        UIError(Error("server not working"));
        return;
    }

    if (soundFileBad == null)
        cacheSoundFiles();



    updateScore(parseFloat(document.getElementById("pronunciation_accuracy").innerHTML));

    document.getElementById("main_title").innerHTML = "Processing new sample...";

    const difficultyValue = document.getElementById('difficultySelect').value;
    sample_difficult = parseInt(difficultyValue);
    switch (sample_difficult) {
        case 0: scoreMultiplier = 1.3; break; // Random
        case 1: scoreMultiplier = 1; break;   // Easy
        case 2: scoreMultiplier = 1.3; break; // Medium
        case 3: scoreMultiplier = 1.6; break; // Hard
        default: scoreMultiplier = 1;
    }

    try {
        await fetch(apiMainPathSample + '/getSample', {
            method: "post",
            body: JSON.stringify({
                "category": sample_difficult.toString(), "language": AILanguage
            }),
            headers: { "X-Api-Key": STScoreAPIKey }
        }).then(res => res.json()).
            then(data => {



                let doc = document.getElementById("original_script");
                currentText = data.real_transcript;
                doc.innerHTML = currentText;

                currentIpa = data.ipa_transcript

                let doc_ipa = document.getElementById("ipa_script");
                doc_ipa.innerHTML = "/ " + currentIpa + " /";

                document.getElementById("recorded_ipa_script").innerHTML = ""
                document.getElementById("pronunciation_accuracy").innerHTML = "";
                document.getElementById("reference_word").innerHTML = "";
                document.getElementById("spoken_word").innerHTML = "";
                document.getElementById("section_accuracy").innerHTML = "Score: " + currentScore.toString() + " - (" + currentSample.toString() + ")";
                currentSample += 1;

                document.getElementById("main_title").innerHTML = page_title;

                document.getElementById("translated_script").innerHTML = data.transcript_translation;

                currentSoundRecorded = false;
                unblockUI();
                document.getElementById("playRecordedAudio").classList.add('disabled');

            })
    }
    catch (error) {
        UIError(error);
    }


};

const updateRecordingState = async () => {
    if (isRecording) {
        isRecording = false;
        stopRecording();
        return
    }
    else {
        isRecording = true;
        await startRecording()
        return;
    }
}



const startRecording = async () => {

    stream = await navigator.mediaDevices.getUserMedia(mediaStreamConstraints);

    mediaRecorder = new MediaRecorder(stream);

    setupMediaRecorderEvents();

    mediaRecorder.start();

    document.getElementById("main_title").innerHTML = "Recording... click again when done speaking";
    document.getElementById("recordIcon").innerHTML = 'pause_presentation';
    blockUI();
    document.getElementById("recordAudio").classList.remove('disabled');
    audioChunks = [];
};

const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
    mediaRecorder = null;

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    document.getElementById("main_title").innerHTML = "Processing audio...";
};

const setupMediaRecorderEvents = () => {
    let currentSamples = 0;
    mediaRecorder.ondataavailable = event => {
        currentSamples += event.data.length;
        audioChunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {

        document.getElementById("recordIcon").innerHTML = 'mic';
        blockUI();


        audioBlob = new Blob(audioChunks, { type: 'audio/ogg;' });

        let audioUrl = URL.createObjectURL(audioBlob);
        audioRecorded = new Audio(audioUrl);

        let audioBase64 = await convertBlobToBase64(audioBlob);

        let minimumAllowedLength = 6;
        if (audioBase64.length < minimumAllowedLength) {
            setTimeout(UIRecordingError, 50); // Make sure this function finished after get called again
            return;
        }

        try {
            // Get currentText from "original_script" div, in case user has change it
            let text = document.getElementById("original_script").innerHTML;
            // Remove html tags
            text = text.replace(/<[^>]*>?/gm, '');
            //Remove spaces on the beginning and end
            text = text.trim();
            // Remove double spaces
            text = text.replace(/\s\s+/g, ' ');
            currentText = [text];

            await fetch(apiMainPathSTS + '/GetAccuracyFromRecordedAudio', {
                method: "post",
                body: JSON.stringify({ "title": currentText[0], "base64Audio": audioBase64, "language": AILanguage }),
                headers: { "X-Api-Key": STScoreAPIKey }

            }).then(res => res.json()).
                then(data => {

                    if (playAnswerSounds)
                        playSoundForAnswerAccuracy(parseFloat(data.pronunciation_accuracy))

                    document.getElementById("recorded_ipa_script").innerHTML = "/ " + data.ipa_transcript + " /";
                    document.getElementById("recordAudio").classList.add('disabled');
                    document.getElementById("main_title").innerHTML = page_title;
                    document.getElementById("pronunciation_accuracy").innerHTML = data.pronunciation_accuracy + "%";
                    document.getElementById("ipa_script").innerHTML = data.real_transcripts_ipa

                    if (!data.is_letter_correct_all_words) {

                    } else {
                        lettersOfWordAreCorrect = data.is_letter_correct_all_words.split(" ")


                        startTime = data.start_time;
                        endTime = data.end_time;


                        real_transcripts_ipa = data.real_transcripts_ipa.split(" ")
                        matched_transcripts_ipa = data.matched_transcripts_ipa.split(" ")
                        wordCategories = data.pair_accuracy_category.split(" ")
                        let currentTextWords = currentText[0].split(" ")

                        coloredWords = "";
                        for (let word_idx = 0; word_idx < currentTextWords.length; word_idx++) {

                            wordTemp = '';
                            for (let letter_idx = 0; letter_idx < currentTextWords[word_idx].length; letter_idx++) {
                                letter_is_correct = lettersOfWordAreCorrect[word_idx][letter_idx] == '1'
                                if (letter_is_correct)
                                    color_letter = 'green'
                                else
                                    color_letter = 'red'

                                wordTemp += '<font color=' + color_letter + '>' + currentTextWords[word_idx][letter_idx] + "</font>"
                            }
                            currentTextWords[word_idx]
                            coloredWords += " " + wrapWordForIndividualPlayback(wordTemp, word_idx)
                        }



                        document.getElementById("original_script").innerHTML = coloredWords

                        currentSoundRecorded = true;
                    }

                    unblockUI();
                    document.getElementById("playRecordedAudio").classList.remove('disabled');

                });
        }
        catch (error) {
            UIError(error);
        }
    };
};

const generateWordModal = (word_idx) => {

    document.getElementById("reference_word").innerHTML = wrapWordForPlayingLink(real_transcripts_ipa[word_idx], word_idx, false, getThemeColor('--text'));

    document.getElementById("spoken_word").innerHTML = wrapWordForPlayingLink(matched_transcripts_ipa[word_idx], word_idx, true, accuracy_colors[parseInt(wordCategories[word_idx])]);
}


const changeLanguage = (language, generateNewSample = false) => {
    voices = synth.getVoices();
    AILanguage = language;
    languageFound = false;
    let languageIdentifier, languageName;
    switch (language) {
        case 'de':
            languageIdentifier = 'de';
            languageName = 'Anna';
            break;

        case 'en':
            languageIdentifier = 'en';
            languageName = 'Daniel';
            break;
    };

    for (idx = 0; idx < voices.length; idx++) {
        if (voices[idx].lang.slice(0, 2) == languageIdentifier && voices[idx].name == languageName) {
            voice_synth = voices[idx];
            languageFound = true;
            break;
        }

    }
    // If specific voice not found, search anything with the same language 
    if (!languageFound) {
        for (idx = 0; idx < voices.length; idx++) {
            if (voices[idx].lang.slice(0, 2) == languageIdentifier) {
                voice_synth = voices[idx];
                languageFound = true;
                break;
            }
        }
    }
    if (generateNewSample)
        getNextSample();
}

//################### Speech-To-Score function ########################
const mediaStreamConstraints = {
    audio: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
    // Add these for better Safari compatibility
    channelCount: 1,
    sampleRate: 44100
  }
}


// ################### Audio playback ##################
const playSoundForAnswerAccuracy = async (accuracy) => {

    currentAudioForPlaying = soundFileGood;
    if (accuracy < mediumScoreThreshold) {
        if (accuracy < badScoreThreshold) {
            currentAudioForPlaying = soundFileBad;
        }
        else {
            currentAudioForPlaying = soundFileOkay;
        }
    }
    playback();

}

const playAudio = async () => {

    document.getElementById("main_title").innerHTML = "Generating sound...";
    playWithMozillaApi(currentText[0]);
    document.getElementById("main_title").innerHTML = "Current Sound was played";

};

function playback() {
    const playSound = ctx.createBufferSource();
    playSound.buffer = currentAudioForPlaying;
    playSound.connect(ctx.destination);
    playSound.start(ctx.currentTime)
}


const playRecording = async (start = null, end = null) => {
    blockUI();

    try {
        if (start == null || end == null) {
            endTimeInMs = Math.round(audioRecorded.duration * 1000)
            audioRecorded.addEventListener("ended", function () {
                audioRecorded.currentTime = 0;
                unblockUI();
                document.getElementById("main_title").innerHTML = "Recorded Sound was played";
            });
            await audioRecorded.play();

        }
        else {
            audioRecorded.currentTime = start;
            audioRecorded.play();
            durationInSeconds = end - start;
            endTimeInMs = Math.round(durationInSeconds * 1000);
            setTimeout(function () {
                unblockUI();
                audioRecorded.pause();
                audioRecorded.currentTime = 0;
                document.getElementById("main_title").innerHTML = "Recorded Sound was played";
            }, endTimeInMs);

        }
    }
    catch {
        UINotSupported();
    }
};

const playNativeAndRecordedWord = async (word_idx) => {

    if (isNativeSelectedForPlayback)
        playCurrentWord(word_idx)
    else
        playRecordedWord(word_idx);

    isNativeSelectedForPlayback = !isNativeSelectedForPlayback;
}



const playCurrentWord = async (word_idx) => {

    document.getElementById("main_title").innerHTML = "Generating word...";
    playWithMozillaApi(currentText[0].split(' ')[word_idx]);
    document.getElementById("main_title").innerHTML = "Word was played";
}

// TODO: Check if fallback is correct
const playWithMozillaApi = (text) => {

    if (languageFound) {
        blockUI();
        if (voice_synth == null)
            changeLanguage(AILanguage);

        var utterThis = new SpeechSynthesisUtterance(text);
        utterThis.voice = voice_synth;
        utterThis.rate = 0.7;
        utterThis.onend = function (event) {
            unblockUI();
        }
        synth.speak(utterThis);
    }
    else {
        UINotSupported();
    }
}

const playRecordedWord = (word_idx) => {

    wordStartTime = parseFloat(startTime.split(' ')[word_idx]);
    wordEndTime = parseFloat(endTime.split(' ')[word_idx]);

    playRecording(wordStartTime, wordEndTime);

}

// ############# Utils #####################
const convertBlobToBase64 = async (blob) => {
    return await blobToBase64(blob);
}

const blobToBase64 = blob => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(blob);
    reader.onload = () => resolve(reader.result);
    reader.onerror = error => reject(error);
});

const wrapWordForPlayingLink = (word, word_idx, isFromRecording, word_accuracy_color) => {
    if (isFromRecording)
        return '<span class="ipa-word-right" style="color:' + word_accuracy_color + ';" onclick="playRecordedWord(' + word_idx + ')">' + word + '</span>';
    else
        return '<span class="ipa-word-left" style="color:' + word_accuracy_color + ';" onclick="playCurrentWord(' + word_idx + ')">' + word + '</span>';
}

const wrapWordForIndividualPlayback = (word, word_idx) => {


    return '<a onmouseover="generateWordModal(' + word_idx.toString() + ')" style = " white-space:nowrap; " href="javascript:playNativeAndRecordedWord(' + word_idx.toString() + ')"  >' + word + '</a> '

}

// ########## Function to initialize server ###############
// This is to try to avoid aws lambda cold start 
try {
    fetch(apiMainPathSTS + '/GetAccuracyFromRecordedAudio', {
        method: "post",
        body: JSON.stringify({ "title": '', "base64Audio": '', "language": AILanguage }),
        headers: { "X-Api-Key": STScoreAPIKey }

    });
}
catch { }

const initializeServer = async () => {

    valid_response = false;
    document.getElementById("main_title").innerHTML = 'Initializing server, this may take up to 2 minutes...';
    let number_of_tries = 0;
    let maximum_number_of_tries = 4;

    while (!valid_response) {
        if (number_of_tries > maximum_number_of_tries) {
            serverWorking = false;
            break;
        }

        try {
            await fetch(apiMainPathSTS + '/GetAccuracyFromRecordedAudio', {
                method: "post",
                body: JSON.stringify({ "title": '', "base64Audio": '', "language": AILanguage }),
                headers: { "X-Api-Key": STScoreAPIKey }

            }).then(
                valid_response = true);
            serverIsInitialized = true;
        }
        catch {
            number_of_tries += 1;
        }
    }
}

if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/service-worker.js').catch(err => {
            console.error("ServiceWorker registration failed:", err);
        });
    });
}

