# 🎵 EE200: Audio Fingerprinting 

**Zapptain America Team Project | Signals, Systems & Networks**

A professional, Shazam-style audio identification web application built with Python and Streamlit. This application takes a short, noisy audio clip and accurately identifies the original song by analyzing its acoustic fingerprint against a pre-indexed database.

## Live Demo

🚀 **Try the application here:**  
[EE200 Audio App](https://ee200audioapp-c5ualwpzchsnqcm5hhffvs.streamlit.app/)

## ✨ Features

* **Instant Cloud Matching:** Uses a pre-computed `.pkl` hash dictionary, allowing the app to bypass strict cloud memory limits and identify tracks in milliseconds without loading heavy raw audio files.
* **Visual Proof Dashboard:** Generates dynamic, dark-themed Matplotlib visualizations explaining the math behind the match:
  * Time-frequency Spectrograms
  * Constellation Peak Maps
  * Offset Histograms (The Alignment Spike)
  * Runner-up comparisons showing the false-positive noise floor
* **Batch Processing:** Upload multiple `.wav` or `.mp3` clips at once and download a fully compiled `results.csv` of the predictions.
* **Format Agnostic:** Supports MP3, WAV, FLAC, and OGG uploads up to 200MB.

---

## 🔬 How It Works (The DSP Pipeline)

This application relies on the physical properties of sound and combinatorial mathematics, rather than machine learning, to find exact matches.

### 1. Feature Extraction (Spectrogram to Constellation)
The raw audio is transformed into a dense time-frequency map using the Short-Time Fourier Transform (STFT). To make the fingerprint robust against background noise, EQ changes, and volume differences, we use a 2D maximum filter to strip away the background and isolate only the highest-energy acoustic peaks, forming a sparse "constellation."

### 2. Fingerprint Hashing
A single peak is vulnerable to distortion. Instead, the algorithm anchors to one peak and groups it with neighboring peaks within a specific target zone. It records the frequencies of the peaks and the exact time difference between them. This creates a highly specific "hash" signature that is invariant to time (it works no matter where the clip starts in the song).

### 3. Database Search & The Alignment Spike
The query hashes are compared against the millions of hashes in the indexed database. Every time a hash matches, it casts a "vote" for the relative time offset between the query clip and the original track. 
* Incorrect songs scatter their chance collisions randomly, forming a flat noise floor. 
* The correct song forces thousands of data points to converge at the exact same offset, creating a massive, mathematically undeniable **Alignment Spike**.

---

## 💻 Tech Stack

* **Frontend UI:** Streamlit
* **Audio Processing:** Librosa
* **Signal Mathematics:** SciPy (`scipy.signal`, `scipy.ndimage`)
* **Array Operations:** NumPy, Pandas
* **Data Visualization:** Matplotlib

---

## 🚀 Running the App Locally

If you want to run this application on your local machine and index your own raw audio files:

1. Clone the repository:
   ```bash
   git clone [https://github.com/Amanag185/EE200_Audio_App.git](https://github.com/Amanag185/EE200_Audio_App.git)
   cd EE200_Audio_App
2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Add your music:**
   Create a new folder named *database* inside the project directory. Place your .mp3 or .wav files inside this folder.
4. **Launch the application:**
   ```bash
   streamlit run app.py

---

## 👥 Contributors

* **Aman Agrawal** - [@Amanag185](https://github.com/Amanag185)
* **Shorya Singh Rathore** - [@shorya-IITK](https://github.com/Shorya-IITK)
