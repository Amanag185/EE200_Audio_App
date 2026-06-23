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
# ==============================================================================
# MATCHING LOGIC
# ==============================================================================
def match_query(query_hashes, db_hashes):
    """Finds all matches and ranks them by their alignment spike."""
    matches = {} 
    
    for q_sig, t_query in query_hashes:
        if q_sig in db_hashes:
            for song_name, t_db in db_hashes[q_sig]:
                offset = t_db - t_query
                if song_name not in matches:
                    matches[song_name] = []
                matches[song_name].append(offset)
                
    if not matches:
        return []

    # Score and rank all matches
    scored_matches = []
    for song, offsets in matches.items():
        rounded_offsets = np.round(offsets, decimals=1)
        if len(rounded_offsets) > 0:
            counts = pd.Series(rounded_offsets).value_counts()
            max_count = counts.max()
            scored_matches.append((song, max_count, offsets))
            
    # Sort by highest spike count (highest score first)
    scored_matches.sort(key=lambda x: x[1], reverse=True)
    
    # If the absolute best match is less than 5, the whole clip is just noise
    if not scored_matches or scored_matches[0][1] < 5:
        return []
        
    # Return the full list (winner + runner-ups) so the UI can draw the noise floors!
    return scored_matches

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide")
st.title("🎵 EE200: Audio Fingerprinting (Zapptain America)")

import pickle

st.info("The Simple Music Databse...")

@st.cache_resource(show_spinner=False)
def load_fast_database():
    with open('database_hashes.pkl', 'rb') as f:
        return pickle.load(f)

db_hashes, global_fs = load_fast_database()

# TABS for UI
tab1, tab2 = st.tabs(["🔍 Identify (Single-Clip Mode)", "📁 Batch Mode"])

# --- TAB 1: SINGLE CLIP MODE ---
# --- TAB 1: SINGLE CLIP MODE ---
with tab1:
    st.markdown("### Identify a single clip and view intermediate steps")
    uploaded_file = st.file_uploader("Upload an audio clip", type=['mp3', 'wav', 'flac', 'ogg'])
    st.caption("200MB per file • MP3, WAV, FLAC, OGG")
    
    if uploaded_file is not None:
        if st.button("Identify Song"):
            with st.spinner("Analyzing audio physics..."):
                audio, fs = librosa.load(uploaded_file, sr=global_fs, mono=True)
                
                f, t, Sxx_dB, f_idx, t_idx = extract_peaks(audio, fs)
                query_hashes = generate_hashes(f_idx, t_idx)
                
                valid_matches = match_query(query_hashes, db_hashes)
                
                st.markdown("---")
                if valid_matches:
                    best_match = valid_matches[0]
                    best_song = best_match[0]
                    best_offsets = best_match[2]
                    
                    st.success(f"### 🎉🎊 MATCH FOUND: **{best_song}**")
                    
                    import matplotlib.pyplot as plt
                    
                    # --- STEP 1 & 2: FEATURE EXTRACTION (Side-by-Side) ---
                    st.markdown("---")
                    st.subheader("STEP 1 • FEATURE EXTRACTION")
                    st.markdown("### Spectrogram and Constellation")
                    st.markdown("The clip is converted to a time-frequency map (left). Only the most prominent local maxima are kept to form the constellation (right).")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_spec, ax_spec = plt.subplots(figsize=(5, 4))
                        fig_spec.patch.set_facecolor('#0E1117')
                        ax_spec.set_facecolor('#0E1117')
                        
                        ax_spec.pcolormesh(t, f, Sxx_dB, shading='gouraud', cmap='magma')
                        ax_spec.set_title("Spectrogram", color='white')
                        ax_spec.set_ylabel('Frequency (Hz)', color='gray')
                        ax_spec.set_xlabel('time (s)', color='gray')
                        ax_spec.tick_params(colors='gray')
                        ax_spec.set_ylim(0, 5000)
                        
                        st.pyplot(fig_spec)

                    with col2:
                        fig_const, ax_const = plt.subplots(figsize=(5, 4))
                        fig_const.patch.set_facecolor('#0E1117')
                        ax_const.set_facecolor('#0E1117')
                        
                        ax_const.scatter(t[t_idx], f[f_idx], c='#00FFFF', s=5, alpha=0.8)
                        ax_const.set_title("Constellation Peaks", color='white')
                        ax_const.set_ylabel('Frequency (Hz)', color='gray')
                        ax_const.set_xlabel('time (s)', color='gray')
                        ax_const.tick_params(colors='gray')
                        ax_const.set_ylim(0, 5000)
                        
                        st.pyplot(fig_const)
                    
                   # --- STEP 3: THE PROOF (Winning Spike) ---
                    st.markdown("---")
                    st.subheader("STEP 2 • THE PROOF")
                    st.markdown("### The alignment spike")
                    st.markdown("Every matched hash votes for a time offset. A genuine match makes them converge: **That spike cannot be a coincidence.**")
                    
                    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
                    fig_hist.patch.set_facecolor('#0E1117')
                    ax_hist.set_facecolor('#0E1117')
                    
                    # Draw the histogram
                    n, bins, patches = ax_hist.hist(best_offsets, bins=100, color='#FFA500') 
                    
                    # 1. Get the VISUAL height for the arrow to point to
                    visual_max_height = int(n.max())
                    peak_x = (bins[n.argmax()] + bins[n.argmax()+1]) / 2
                    
                    # 2. Get the ACTUAL algorithmic score for the text label
                    actual_spike_count = best_match[1]

                    # Add the pointing arrow using the true count
                    ax_hist.annotate(
                        f'{actual_spike_count} hashes\nalign here', 
                        xy=(peak_x, visual_max_height), 
                        xytext=(30, -30),
                        textcoords='offset points',
                        arrowprops=dict(color='#FFA500', arrowstyle='->', lw=1.5),
                        color='#FFA500',
                        fontsize=10
                    )
                    
                    ax_hist.set_xlabel("time offset (database frame - query frame)", color='gray')
                    ax_hist.set_ylabel("# hashes", color='gray')
                    ax_hist.tick_params(colors='gray')
                    ax_hist.spines['top'].set_visible(False)
                    ax_hist.spines['right'].set_visible(False)
                    ax_hist.spines['bottom'].set_color('gray')
                    ax_hist.spines['left'].set_color('gray')
                    
                    st.pyplot(fig_hist)

                    # --- STEP 4: RUNNER-UPS (Failed Matches) ---
                    if len(valid_matches) > 1:
                        st.markdown("---")
                        st.markdown("#### Runner-Up Comparisons (False Positives)")
                        st.markdown("Notice how incorrect songs just produce a flat noise floor of random chance matches, without a unified spike.")
                        
                        runner_ups = valid_matches[1:4]
                        cols = st.columns(len(runner_ups))
                        
                        for i, col in enumerate(cols):
                            with col:
                                r_song, r_count, r_offsets = runner_ups[i]
                                
                                fig_r, ax_r = plt.subplots(figsize=(4, 3))
                                fig_r.patch.set_facecolor('#0E1117')
                                ax_r.set_facecolor('#0E1117')
                                
                                ax_r.hist(r_offsets, bins=100, color='gray', alpha=0.7)
                                ax_r.set_title(r_song, color='gray', fontsize=10)
                                ax_r.tick_params(colors='gray', labelsize=8)
                                ax_r.spines['top'].set_visible(False)
                                ax_r.spines['right'].set_visible(False)
                                
                                st.pyplot(fig_r)
                     # --- STEP 5: THE FINAL VERDICT ---
                    st.markdown("---")
                    st.subheader("📝 The Final Verdict")
                    
                    # Pull the exact number of matching hashes from your scoring function
                    winning_spike_count = best_match[1]
                    
                    st.info(f"""
                    **Mathematical Confirmation:** The algorithm detected exactly **{winning_spike_count}** constellation hashes from your uploaded clip that aligned flawlessly with the database track for **{best_song}**. 
                    
                    Because random audio noise only produces scattered, coincidental matches of 1 to 4 hashes (as seen in the runner-up graphs), an alignment spike of **{winning_spike_count}** hashes represents an exact, undeniable acoustic fingerprint match.
                    """)    
                else:
                    st.error("### ❌ NO MATCH FOUND in Database")

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
