/**
 * Bengaluru Road Accident Risk Prediction and Safety Recommendation System
 * Script: static/js/voice.js
 * 
 * Browser Web Speech API Integration:
 * - Speech-to-Text (webkitSpeechRecognition / SpeechRecognition)
 * - Text-to-Speech (speechSynthesis)
 * - Graceful fallback when browser speech API is unavailable
 */

class VoiceAssistant {
    constructor() {
        this.recognition = null;
        this.synthesis = window.speechSynthesis || null;
        this.isRecording = false;
        this.initSpeechRecognition();
    }

    initSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            this.recognition = new SpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = 'en-IN'; // Indian English
        } else {
            console.warn('[VOICE] Web Speech API (SpeechRecognition) is not supported in this browser.');
        }
    }

    startListening(targetInputId, micButtonId, onCompleteCallback) {
        if (!this.recognition) {
            alert('Voice recognition (Speech-to-Text) is not supported by your current browser. Please type your query.');
            return;
        }

        const inputEl = document.getElementById(targetInputId);
        const micBtn = document.getElementById(micButtonId);

        if (this.isRecording) {
            this.recognition.stop();
            return;
        }

        this.recognition.onstart = () => {
            this.isRecording = true;
            if (micBtn) {
                micBtn.classList.add('btn-mic-active');
                micBtn.innerHTML = '<i class="fa-solid fa-microphone-lines"></i> Listening...';
            }
        };

        this.recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            if (inputEl) {
                inputEl.value = transcript;
            }
            if (onCompleteCallback) {
                onCompleteCallback(transcript);
            }
        };

        this.recognition.onerror = (event) => {
            console.error('[VOICE ERROR]', event.error);
            this.resetMicButton(micBtn);
        };

        this.recognition.onend = () => {
            this.isRecording = false;
            this.resetMicButton(micBtn);
        };

        try {
            this.recognition.start();
        } catch (err) {
            console.error('[VOICE ERROR] Failed to start:', err);
            this.resetMicButton(micBtn);
        }
    }

    resetMicButton(micBtn) {
        if (micBtn) {
            micBtn.classList.remove('btn-mic-active');
            micBtn.innerHTML = '<i class="fa-solid fa-microphone"></i>';
        }
    }

    speak(text, buttonEl = null) {
        if (!this.synthesis) {
            console.warn('[VOICE] Speech Synthesis is not supported in this browser.');
            return;
        }

        // If speaking, stop it (toggle behavior)
        if (this.synthesis.speaking) {
            this.synthesis.cancel();
            if (buttonEl) buttonEl.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
            return;
        }

        // Clean markdown syntax before speaking
        const cleanText = text
            .replace(/[#*_`~-]/g, '')
            .replace(/\[.*?\]\(.*?\)/g, '')
            .replace(/\n+/g, '. ');

        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.lang = 'en-IN';
        utterance.rate = 1.0;
        utterance.pitch = 1.0;

        if (buttonEl) {
            buttonEl.innerHTML = '<i class="fa-solid fa-stop text-danger"></i>';
            utterance.onend = () => {
                buttonEl.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
            };
            utterance.onerror = () => {
                buttonEl.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
            };
        }

        this.synthesis.speak(utterance);
    }
}

// Global instance
window.voiceAssistant = new VoiceAssistant();
