import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter
import librosa
import pandas as pd
import os
import glob

# ==============================================================================
# CORE ALGORITHM (From Q3A)
# ==============================================================================
@st.cache_data(show_spinner=False)
def extract_peaks(audio, fs, nperseg=1024, filter_size=10):
    f, t, Sxx = signal.spectrogram(audio, fs, nperseg=nperseg, noverlap=nperseg//2)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)
    local_max = maximum_filter(Sxx_dB, size=filter_size) == Sxx_dB
    threshold_db = Sxx_dB.max() - 20 
    peaks = local_max & (Sxx_dB > threshold_db)
    freq_idx, time_idx = np.where(peaks)
    return f, t, Sxx_dB, freq_idx, time_idx

@st.cache_data(show_spinner=False)
def generate_hashes(freq_idx, time_idx, target_zone_width=5):
    hashes = []
    num_peaks = len(time_idx)
    sorted_indices = np.argsort(time_idx)
    t_sorted, f_sorted = time_idx[sorted_indices], freq_idx[sorted_indices]
    
    for i in range(num_peaks):
        t1, f1 = t_sorted[i], f_sorted[i]
        for j in range(1, target_zone_width + 1):
            if i + j < num_peaks:
                t2, f2 = t_sorted[i + j], f_sorted[i + j]
                delta_t = t2 - t1
                if delta_t > 0:
                    hashes.append(((f1, f2, delta_t), t1))
    return hashes

# ==============================================================================
# DATABASE INDEXING
# ==============================================================================
@st.cache_resource(show_spinner=True)
def build_database(database_folder="database"):
    """Loads all songs in the database folder and creates a master hash dictionary."""
    db_hashes = {}
    
    if not os.path.exists(database_folder):
        os.makedirs(database_folder)
        
    song_files = glob.glob(os.path.join(database_folder, "*.mp3")) + glob.glob(os.path.join(database_folder, "*.wav"))
    
    for file in song_files:
        song_name = os.path.splitext(os.path.basename(file))[0]
        # Fast load: 11kHz, first 60 seconds is enough to build a solid fingerprint
        audio, fs = librosa.load(file, sr=11025, mono=True, duration=60)
        _, _, _, f_idx, t_idx = extract_peaks(audio, fs)
        hashes = generate_hashes(f_idx, t_idx)
        
        # Store in dict: {hash_signature: [(song_name, time_offset), ...]}
        for h_sig, t_db in hashes:
            if h_sig not in db_hashes:
                db_hashes[h_sig] = []
            db_hashes[h_sig].append((song_name, t_db))
            
    return db_hashes, fs

# ==============================================================================
# MATCHING LOGIC
# ==============================================================================
def match_query(query_hashes, db_hashes):
    """Finds the song with the highest alignment spike."""
    matches = {} # Format: {song_name: [offset1, offset2, ...]}
    
    for q_sig, t_query in query_hashes:
        if q_sig in db_hashes:
            for song_name, t_db in db_hashes[q_sig]:
                offset = t_db - t_query
                if song_name not in matches:
                    matches[song_name] = []
                matches[song_name].append(offset)
                
    if not matches:
        return None, None

    # Find the song with the largest identical offset spike
    best_song = None
    best_spike_count = 0
    best_offsets = []

    for song, offsets in matches.items():
        # Round offsets to group slight variations
        rounded_offsets = np.round(offsets, decimals=1)
        if len(rounded_offsets) > 0:
            # Count the most frequent offset
            counts = pd.Series(rounded_offsets).value_counts()
            max_count = counts.max()
            if max_count > best_spike_count:
                best_spike_count = max_count
                best_song = song
                best_offsets = offsets # Keep raw offsets for plotting
                
    # If the highest spike is less than 5, it's probably random noise, not a match
    if best_spike_count < 5:
        return None, None
        
    return best_song, best_offsets

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide")
st.title("🎵 EE200: Audio Fingerprinting (Zapptain America)")

import pickle

st.info("Loading Fast Database...")

@st.cache_resource(show_spinner=False)
def load_fast_database():
    with open('database_hashes.pkl', 'rb') as f:
        return pickle.load(f)

db_hashes, global_fs = load_fast_database()

# TABS for UI
tab1, tab2 = st.tabs(["🔍 Identify (Single-Clip Mode)", "📁 Batch Mode"])

# --- TAB 1: SINGLE CLIP MODE ---
with tab1:
    st.markdown("### Identify a single clip and view intermediate steps")
    uploaded_file = st.file_uploader("Upload an audio clip", type=['mp3', 'wav', 'flac', 'ogg'])
    
    if uploaded_file is not None:
        if st.button("Identify Song"):
            with st.spinner("Analyzing audio physics..."):
                # Load Query
                audio, fs = librosa.load(uploaded_file, sr=global_fs, mono=True)
                
                # 1. Spectrogram & Peaks
                f, t, Sxx_dB, f_idx, t_idx = extract_peaks(audio, fs)
                query_hashes = generate_hashes(f_idx, t_idx)
                
                # 2. Match
                best_song, offsets = match_query(query_hashes, db_hashes)
                
                # --- VISUALIZATION OUTPUT ---
                st.markdown("---")
                if best_song:
                    st.success(f"### 🎉🎊 MATCH FOUND: **{best_song}**")
                else:
                    st.error("### ❌ NO MATCH FOUND in Database")
                   
                    import matplotlib.pyplot as plt
                    st.markdown("---")
                    st.subheader("STEP 1 • THE PROOF")
                    st.markdown("### The alignment spike")
        
       
                    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
                    fig_hist.patch.set_facecolor('#0E1117')
                    ax_hist.set_facecolor('#0E1117')
        
        # NOTE: Make sure 'offsets' is the actual name of your time-difference variable!
                    ax_hist.hist(offsets, bins=100, color='#FFA500') 
        
                    ax_hist.set_xlabel("time offset (database frame - query frame)", color='gray')
                    ax_hist.set_ylabel("# hashes", color='gray')
                    ax_hist.tick_params(colors='gray')
                    ax_hist.spines['top'].set_visible(False)
                    ax_hist.spines['right'].set_visible(False)
                    ax_hist.spines['bottom'].set_color('gray')
                    ax_hist.spines['left'].set_color('gray')
        
                    st.pyplot(fig_hist)

        

        # 2. The Constellation
        st.markdown("---")
        st.subheader("STEP 2 • FEATURE EXTRACTION")
        
        fig_const, ax_const = plt.subplots(figsize=(10, 4))
        fig_const.patch.set_facecolor('#0E1117')
        ax_const.set_facecolor('#0E1117')
        
        # NOTE: Make sure 't_peaks' and 'f_peaks' match your variable names!
        ax_const.scatter(t_peaks, f_peaks, c='#00FFFF', s=5, alpha=0.8)
        ax_const.set_ylabel('Frequency (Hz)', color='gray')
        ax_const.set_xlabel('time (s)', color='gray')
        ax_const.tick_params(colors='gray')
        ax_const.set_ylim(0, 5000)
        
        st.pyplot(fig_const)

                st.markdown("#### Step 1: Spectrogram & Constellation")
                fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
                ax1.pcolormesh(t, f, Sxx_dB, shading='gouraud', cmap='magma')
                ax1.set_title("Query Spectrogram")
                ax1.set_ylim(0, 5000)
                
                ax2.scatter(t[t_idx], f[f_idx], c='cyan', s=10)
                ax2.set_facecolor('#1e1e1e')
                ax2.set_title("Constellation Peaks")
                ax2.set_ylim(0, 5000)
                st.pyplot(fig1)

                if offsets:
                    st.markdown("#### Step 2: Hash Alignment (The Proof)")
                    fig2, ax3 = plt.subplots(figsize=(10, 3))
                    ax3.hist(offsets, bins=100, color='orange', edgecolor='black')
                    ax3.set_title(f"Offset Histogram for {best_song}")
                    st.pyplot(fig2)

# --- TAB 2: BATCH MODE ---
with tab2:
    st.markdown("### Identify multiple clips at once")
    batch_files = st.file_uploader("Upload multiple query clips", type=['wav', 'mp3'], accept_multiple_files=True, key="batch")
    
    if batch_files:
        if st.button("Run Batch Processing"):
            results = []
            
            # Progress bar for batch mode
            progress_bar = st.progress(0)
            for i, file in enumerate(batch_files):
                audio, fs = librosa.load(file, sr=global_fs, mono=True)
                _, _, _, f_idx, t_idx = extract_peaks(audio, fs)
                query_hashes = generate_hashes(f_idx, t_idx)
                
                best_song, _ = match_query(query_hashes, db_hashes)
                prediction = best_song if best_song else "None"
                
                results.append({"filename": file.name, "prediction": prediction})
                progress_bar.progress((i + 1) / len(batch_files))
                
            # Create CSV
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download results.csv",
                data=csv,
                file_name='results.csv',
                mime='text/csv',
            )
