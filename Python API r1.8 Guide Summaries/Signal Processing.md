Signal Processing (contrib)
* tf.contrib.signal

This module is used to process primitives. Useful for models handeling audio though used in many domains.
To handel variable length signals you can frame them into multople fixed length windows. Windows may overlap if a step is less than the frame's length.

* tf.contrib.signal.frame
This does the above. 

	# batch of f32 time domain signals in [-1, 1] with shape [batch_size, signal_length]. Both of the shapes may be unknown.
	signals = tf.placeholder(tf.float32,[none,none])
	
	# compute [batch_size, ? , 128] tensor of fixed length with overlapping windows where each overlap is 75% of the previous(frame_length - frame_step samples of overlap).
	frames = tf.contrib.signal.frame(signals, frame_length=128, frame_stap=32)

* tf.contrib.signal.frame
This axis parameter gives you the ability to frame tensors with an inner structure(spectrogram for example)

	# `magnitude_spectrograms` is a [batch_size, ? , 129] tensor of spectrograms. We would like to produce overlapping fixed-size spectrogram patches; for example for use in a 
	# situation where a fixed size input is needed.
	magnitude_spectrograms = tf.abs(tf.contrib.signal.stft(signals, frame_length=256, frame_stem=64, fft_length=256))
	# `spectrogram_patches` is a [batch_size, ?, 
	# 64,129] tensor containing a variable number of [64,129] specrogramn patches per batch item.
	spectrogram_patches = tf.contrib.signal.frame(magnitide_spectrograms, frame_length=64, frame_step=16, axis=1)
	
Reconstructing frame sequences and applying a tapering windows
* tf.contrib.signal.overlap_and_add
Used to reconstruct a signal from a framed frepresentation. The following code reconstruct the signal produced in the preceding example:
	# Reconstructs a `signals` from `frames` produced in teh above hoever the magnitude of the `reconstructed_signal` will be greater than the `signals`.
	reconstructd_signals = tf.contrib.signal.overlap_and_add(frames, frame_step=32)
	
	note that because frame_step is 25% frame_length the construction will have a greate mag than the orginal signal. To compensate wie use tapering window function. 
	If thewindow function satistisfies the constand overlap0Add((COLA) property for a given frame step, then it will recover the origonal signals.

* tf.contrib.signal.hamming_window
* tf.contrib.signal.hann_window

These two satisfy the COLA property for a 75% overlap
	frame_length = 128 
	frame_step = 32
	windowed_frames = frames* tf.contrib.signal.hann_window(frame_lenght)
	reconstructed_signals = tf.contrib.signal.overla_and_add(windowed_frames, frame_step)

Computeing spectrograms
A spectrogram is a time frequency decomposition of a signals that indiicates it's frequency content over time. The most common approach to computing sprograms is to take the magnitude of 
the short time fourier transform(STFT) which tf.contrib.signal.stft can compute:
	# A batch of f32 time domain signals is in the range [-1.1] with the shape of [batch_size, signal_length]. both the batch_size and the signal)length may be unknown
	signals = tf. placeholder(tf.float32,[none,none])
	#the stfts is a complex64 tensor representing the STFT of each signal in signals. It's shape is [batch_size, ?, fft_unique_bins] where the fft_unique_bins = fft_length // 2 + 1 = 513
	stfts = tf.contrib.signal.stft(signals, frame_length=1024, frame_step=512,fft_length=1024)
	#power spectrogram is the squared magnitude of the compex valued STFT. A f32 tensor of shape[batch_size, ?, 513]
	power_spectrograms = tf.real(stfts * tf.conj*stfts))
	#an energy spectrogram is the mag of the complex valued STFT# A f32 tensor of shape[batch)size, ? , 513]
	magnitude_spectrograms = tf.abs(stfts)
You may use a power spectrogram or a magnitude spectrogram. Each has advateds note that if you apply a logarithmic compression. The power spectro and magnitide differ by factor of 2

Logarithmic compression
Common to apply a compressive nonlinearity such as the logarithm or pwoer law compression to spectrograms. Helps balance importance of detail in spectrum to match human sensitivity.
	log_offset = 1e=6
	log_magnitude_specrograms = tf.log(magnitude_spectrograms + log_offset)
	
Computing log-mel spectrograms
mel scale is common rewieighting of audio frenquency dimensions.
* tf.contrib.signal.linear_to_mel_weight_matrix
produces a matrix you can use to convert a spectrogram to the mel scale.

Computing Mel-Frequency Cepstral Coefficients (MFCCs)
Call
* tf.contrib.signal.mfccs_from_log_mel_spectrograms 
to compute MFCCs
