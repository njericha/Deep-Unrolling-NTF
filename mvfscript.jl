#=
Scripts used to plot the output to seperation tests in mvf.jl
=#

using Plots
using WAV

include("./mvf.jl")
include("./synthetic_music.jl")

# Generate example song spectogram
(Y, Φ, Xs, xs, D, spectrums, t, times, midi_notes, freqs) = make_spectogram();
DD = double_tensor(D);
# t is the finely sampled times in the audio domain
# times is the corse sample times in the STFT space

# Perform the factorization Y = A*DD*b = [A1; A2] * DD * [b1; b2]
# Equivilently Y = A1*D*b1 + A2*D*b2
(A1, A2, b1, b2, error, norm_grad_A, norm_grad_b) = nnls_seperate(
    Y', DD; maxiter=25, tol=1e-3, λA=1e-8, ϵA=5e0, γA=4e1, μb=1.5e3, δb=0);

# Create Estimates of the sources in STFT-space
X1_estimate, X2_estimate = (A1*(D×₃b1))', (A2*(D×₃b2))';
X1, X2 = Xs;

"""
    align_sources!((A1, b1, X1_estimate), (A2, b2, X2_estimate), X1, X2)

swaps (A1, b1, X1_estimate) and (A2, b2, X2_estimate) if it better matches
the true spectrums X1 and X2
"""
function align_sources!((A1, b1, X1_estimate), (A2, b2, X2_estimate), X1, X2)
    function swap!(x, y)
        temp = copy(x)
        x .= y; y.= temp
    end
    if norm(X1_estimate - X1) > norm(X1_estimate - X2)
        swap!.((A1, b1, X1_estimate), (A2, b2, X2_estimate))
    end
end
align_sources!((A1, b1, X1_estimate), (A2, b2, X2_estimate), X1, X2);

#= Evaluations =#
myerror(xhat, x) = (norm(xhat - x) / norm(x))^2 # relative mean squared error

@show myerror(X1_estimate, X1)
@show myerror(X2_estimate, X2)
@show myerror(X1_estimate + X2_estimate, Y)
@show myerror(b1, spectrums[1])
@show myerror(b2, spectrums[2])

using Statistics: mean
rms(x) = sqrt(mean(x .^ 2))
snr(xhat, x) = 20*log10(rms(x) / rms(x - xhat))

@show snr(X1_estimate, X1)
@show snr(X2_estimate, X2)
@show snr(X1_estimate + X2_estimate, Y)
@show snr(b1, spectrums[1])
@show snr(b2, spectrums[2])

#= Smart Rescaling for X estimates=#
c1 = find_best_scale(X1_estimate, X1)
c2 = find_best_scale(X2_estimate, X2)
X1_estimate .*= c1
X2_estimate .*= c2

@show myerror(X1_estimate, X1)
@show myerror(X2_estimate, X2)
@show myerror(X1_estimate + X2_estimate, Y)

@show snr(X1_estimate, X1)
@show snr(X2_estimate, X2)
@show snr(X1_estimate + X2_estimate, Y)

#= Plots =#
# learned spectrums b1 and b2 with the ground truth
plot([b1, b2],
    title="Learned vs True spectrums",
    xlabel="harmonic #",
    ylabel="relative amplitude",
    label=["learned spectrum 1" "learned spectrum 2"]
)
plot!(spectrums, label=["true spectrum 1" "true spectrum 2"])

# Learned notes
heatmap(times, midi_notes, A1',
    title="Learned Notes for source 1",
    xaxis="time (s)",
    yaxis="note (MIDI number)"
)

heatmap(times, midi_notes, A2',
    title="Learned Notes for source 2",
    xaxis="time (s)",
    yaxis="note (MIDI number)"
)

# Learned vs True Instrument Spectrums absolute difference
heatmap(times, freqs, Y,
title= "Input Spectogram",
xaxis="time (s)",
ylabel="frequency (Hz)")

heatmap(times, freqs, abs.(X1_estimate - X1),
title= "Absolute Difference Between Learned and True\nSpectograms for Source 1\n",
xaxis="time (s)",
ylabel="frequency (Hz)")

heatmap(times, freqs, abs.(X2_estimate - X2),
title= "Absolute Difference Between Learned and True\nSpectograms for Source 2\n",
xaxis="time (s)",
ylabel="frequency (Hz)")

heatmap(times, freqs, X1_estimate,
title= "Learned Spectogram for Source 1",
xaxis="time (s)",
ylabel="frequency (Hz)")
heatmap(times, freqs, X1,
title= "True Spectogram for Source 1",
xaxis="time (s)",
ylabel="frequency (Hz)")

heatmap(times, freqs, X2_estimate,
title= "Learned Spectogram for Source 2",
xaxis="time (s)",
ylabel="frequency (Hz)")
heatmap(times, freqs, X2,
title= "True Spectogram for Source 2",
xaxis="time (s)",
ylabel="frequency (Hz)")

#= Recover time domain signal =#
x1, x2 = xs
chopsize = 20 #removes end effects
timedomain(X) = myistft(X .* cis.(Φ))[begin+chopsize:end-chopsize]

x1_estimate, x2_estimate, x1_theory, x2_theory = [(timedomain(X)) for
    X ∈ (X1_estimate, X2_estimate, X1, X2)]; #normalized

# Chop t to the same length as the time domain signals, centered on the interval
choppedlength = length(t) - length(x1_estimate)
tchop = t[begin + (choppedlength+1)÷2 : end - choppedlength÷2]
sample_rate = Integer((t[2]-t[1])^(-1))

#= Plot time doamin signals =#
plot(t, x1,
title="True Source 1",
xaxis = "time (s)",
yaxis = "signal")
plot(tchop, x1_theory,
title="Best Source 1 Recovery",
xaxis = "time (s)",
yaxis = "signal")
plot(tchop, x1_estimate,
title="Source 1 Estimate",
xaxis = "time (s)",
yaxis = "signal")

plot(t, x2,
title="True Source 2",
xaxis = "time (s)",
yaxis = "signal")
plot(tchop, x2_theory,
title="Best Source 2 Recovery",
xaxis = "time (s)",
yaxis = "signal")
plot(tchop, x2_estimate,
title="Source 2 Estimate",
xaxis = "time (s)",
yaxis = "signal")


#= Play time domain signals =#
wavplay(x1, sample_rate)
wavplay(x1_theory, sample_rate)
wavplay(x1_estimate, sample_rate)

wavplay(x2, sample_rate)
wavplay(x2_theory, sample_rate)
wavplay(x2_estimate, sample_rate)

wavplay(x1 + x2, sample_rate)
wavplay(x1_estimate + x2_estimate, sample_rate)

@show snr(x1_estimate, x1_theory) # same as in STFT space by Parseval's theorem
@show snr(x2_estimate, x2_theory)
@show snr(x1_estimate + x2_estimate, x1_theory + x2_theory)
