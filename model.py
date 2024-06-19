import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import sounddevice as sd
import numpy as np
import librosa
import joblib

def record_audio(duration, sample_rate=44100):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()
    print("Recording finished.")
    return audio.flatten(), sample_rate

def extract_features(audio_data, sample_rate=44100):
    # Extracting pitch (fundamental frequency)
    pitches, magnitudes = librosa.core.piptrack(y=audio_data, sr=sample_rate)
    
    # Only keep the highest pitch at each frame
    frequencies = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            frequencies.append(pitch)
    
    frequencies = np.array(frequencies)
    
    # Extract features
    features = {
        'MDVP:Fo(Hz)': np.mean(frequencies),
        'MDVP:Fhi(Hz)': np.max(frequencies),
        'MDVP:Flo(Hz)': np.min(frequencies),
        'MDVP:Jitter(%)': np.std(frequencies) / np.mean(frequencies) * 100 if np.mean(frequencies) != 0 else 0,
        'MDVP:Jitter(Abs)': np.std(frequencies),
        # Placeholder values for other features that need more complex extraction
        'MDVP:RAP': 0,
        'MDVP:PPQ': 0,
        'Jitter:DDP': 0,
        'MDVP:Shimmer': 0,
        'MDVP:Shimmer(dB)': 0,
        'Shimmer:APQ3': 0,
        'Shimmer:APQ5': 0,
        'MDVP:APQ': 0,
        'Shimmer:DDA': 0,
        'NHR': 0,
        'HNR': 0,
        'RPDE': 0,
        'DFA': 0,
        'spread1': 0,
        'spread2': 0,
        'D2': 0,
        'PPE': 0
    }
    
    return features

def main():
    # Load and prepare the data
    data = pd.read_csv('pd_vocal_frequencies.csv')
    X = data.drop(columns=['name', 'status'])
    y = data['status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'pd_diagnosis_model.pkl')

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Load the trained model for testing new input
    model = joblib.load('pd_diagnosis_model.pkl')

    # Display the motivational quote for the user to read
    print("\nPlease read the following quote aloud while recording your voice:")
    print("Today is going to be a very good day for me! \n")

    # Record a new voice sample for testing
    duration = 5  # seconds
    new_audio_data, sample_rate = record_audio(duration)
    
    # Extract features from the new audio data
    new_audio_features = extract_features(new_audio_data, sample_rate)
    new_audio_features_df = pd.DataFrame([new_audio_features])

    # Predict the diagnosis for the new audio sample
    diagnosis_prediction = model.predict(new_audio_features_df)
    print("Predicted Diagnosis for the new audio sample:", 'PD' if diagnosis_prediction[0] == 1 else 'Non-PD')

if __name__ == "__main__":
    main()
