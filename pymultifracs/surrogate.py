def create_surrogate(mrq, j1, j2, n_surrogate=1):
    """
    WIP: needs SWT/CWT-based MFA or NUFFT
    """
    
    idx_nonan = {
        j: (~np.isnan(mrq.values[j])).all(axis=1)
        for j in range(j1, j2+1)
    }
    
    N_fft = {j: idx_nonan[j2].sum() * 2 ** (j2-j) for j in range(j1, j2+1)}

    X_fft = np.zeros((N_fft[j1], mrq.n_rep, j2-j1+1), dtype='cdouble')

    # freqs=fftfreq(N_fft[j2], 2) 
    # print(freqs)
    # print(np.r_[freqs[:N_fft[j2]//2], freqs[-N_fft[j2]//2:]])
    # print(np.r_[freqs[:N_fft[j2]//2]])
    # print(np.r_[freqs[-N_fft[j2]//2:]])
    
    for idx_j, j in enumerate(range(j1, j2+1)):

        fft_ = fft(mrq.values[j][idx_nonan[j]], n=N_fft[j], axis=0)
        X_fft[:N_fft[j]//2, :, idx_j] = fft_[:N_fft[j]//2]
        X_fft[-N_fft[j]//2:, :, idx_j] = fft_[N_fft[j]//2:]

        # print(N_fft[j], fft_.shape)

    mag = np.abs(X_fft)
    phase = np.angle(X_fft)

    rng = np.random.default_rng()
    
    angle = rng.uniform(0, 2*np.pi, (N_fft[j1], n_surrogate))
    # angle = np.zeros_like(angle)
    
    new_values = {
        j: np.zeros((*mrq.values[j].shape, n_surrogate)) + np.nan
        for j in range(j1, j2+1)
    }

    # print(idx_nonan[j1].sum())

    # print(ifft(
    #         mag[:, :, None, idx_j] + np.exp(1j * (phase[:, :, None, idx_j] + angle[:, None, :])),
    #         axis=0, n=idx_nonan[j].sum()).real.shape)
    # print(new_values[j][idx_nonan[j], :].shape)
    
    for idx_j, j in enumerate(range(j1, j2+1)):

        # mag_j = np.zeros((N_fft[j] * 2, mrq.n_rep, 1))

        # mag_j[:N_fft[j]//2] = mag[:N_fft[j] // 2]
        # mag_j[-N_fft[j]//2:] = mag[:N_fft[j] // 2]

        # phase_j[
        
        mag_j = np.r_[mag[:N_fft[j]//2, :, None, idx_j], mag[-N_fft[j]//2:, :, None, idx_j]]

        phase_j = phase[:, :, None, idx_j] + angle[:, None, :]
        phase_j = np.r_[phase_j[:N_fft[j]//2], phase_j[-N_fft[j]//2:]]

        # print(mag_j.shape, phase_j.shape, angle.shape)
        coefs = mag_j + np.exp(1j * phase_j)
        assert coefs.shape[0] == N_fft[j]

        # print(ifft(coefs, axis=0, n=N_fft[j]).real)
        idx_nonan[j][N_fft[j]+2:] = False
        new_values[j][idx_nonan[j], :] = ifft(coefs, axis=0, n=N_fft[j]).real
        new_values[j] = new_values[j].reshape(mrq.values[j].shape[0], -1)
        # new_values[j] = new_values[j][3:]
    
    return mrq._from_dict({
        'values': new_values,
    })