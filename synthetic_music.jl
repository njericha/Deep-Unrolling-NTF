"""
Generates synthetic music data to use NMF on
"""

using Random
using STFT

"""
# Inputs
- t is the time points to create the envelope on
- delay is the time value where the envelope starts

# Parameters
- a is the attack or onset duration
- s is the note duration aka sustain
- r is the release or offset duration

"a" must be <= "s"
"""
function envelope(t, delay; ϵ = 0, asr=(0.01,0.35,0.1))
    #ϵ = 1e-4 #prevent divide by zero errors
    d=delay
    a,s,r = asr
    A = @. -exp(-a)/a * (t-d) * (t >= d) * (t <= d+a)
    S = @. exp(-(t - d)) * (t >= d+a) * (t <= d+s)
    R = @. -exp(-s)/r * (t-s-d-r) * (t >= d+s) * (t <= d+s+r)
    env = A + S + R
    return @. env + (ϵ * (t < d))
end

# ensures amplitude normalization in slightly under 1 to avoid speaker clipping issues
normalized(y; ceiling = 0.98) =  y ./ maximum(abs.(y)) .* ceiling

"""
Time points t, the pitch is the fundimental frequency, harmonics is a list giving the relative weights of each harmonic 
"""
function note(t, pitch, harmonics)
    f₀ = pitch
    ϕ = 0#rand()*2π #global phase shift
    y = sum(b .* sin.(2π*f₀*t*n .+ ϕ) for (n, b) ∈ enumerate(harmonics))
    return normalized(y)
end

function hann(N::Int)
    N = N - N % 2 #makes sure N even
    n = 0:N
    return @. sin(π*n/N)^2
end

"""
notename_to_keynumber convers a string to an integer corresponding to the midi number
# Examples
notename_to_keynumber("A4") = 49
notename_to_keynumber("B4") = 51
notename_to_keynumber("A#4") = 50
notename_to_keynumber("A♯4") = 50
notename_to_keynumber("Bb4") = 50
notename_to_keynumber("B♭4") = 50

"""
function notename_to_keynumber(s::String) #TODO convert this to a type?
    accidental = " "

    letters = "C D EF G A B" #spaces are for accidentals

    if length(s) == 2
        letter, octave = s
    elseif length(s) == 3
        letter, accidental, octave = s
        if !occursin(accidental,"#♯b♭")
            throw(ArgumentError("Invalid accidental symbol \"$accidental\""))
        end
    else
        throw(ArgumentError("length of s must be 2 or 3"))
    end

    if !occursin(letter,letters)
        throw(ArgumentError("Invalid letter name \"$letter\""))
    end

    offset = 9 # since A0 needs to return 1
    key = 12*parse(Int, octave) + findfirst(letter, letters) - offset

    if occursin(accidental, "#♯")
        key += 1
    elseif occursin(accidental, "b♭")
        key -= 1
    end
    
    return key
end

keynumber_to_frequency(n::Integer) = @. 440*2^((n-49)/12)
notename_to_frequency(s::String) = s |> notename_to_keynumber |> keynumber_to_frequency

"""
make a time series x at time samples t with a given:
- frequency spectrum ex. [1,1/2,1/4]
- notes ex. ["C4", "G3"]
- delays ex. [0.2, 0.6]
- attack-sustain-release ex. (0.02,0.3,0.1)
"""
function make_source(t, spectrum, notes, delays, asr=(0.02,0.3,0.1))
    x_heldnotes = [note(t, pitch, spectrum) for pitch ∈ notename_to_frequency.(notes)]
    x_envelopes = [envelope(t, delay, asr=asr) for delay ∈ delays]
    x = normalized(sum(note .* env for (note, env) ∈ zip(x_heldnotes, x_envelopes)))
    return x
end

function close_in_cents(f,g;tol=20) #within tol cents
	#f_cents = 1200*log2(f/440)
	#g_cents = 1200*log2(g/440)
	#return abs(f_cents - g_cents) < tol
	return 1200*abs(log2(f/g)) < tol #1200 cents per octave (power of 2)
end

function close_in_frequency(f,g;tol=10) #within tol Hz
	return abs(f-g) < tol #1200 cents per octave (power of 2)
end


"""
makes the song by combining two sources.
N is the number of harmonics which is left as a parameter
so it can be set external to the function
"""
function make_sources(t, N)

    notes1 = ["C3", "F3", "E3", "G2", "C3"]
    notes2 = ["G2", "C3", "G2", "B3"]
    delays1 = [0, 0.4, 0.8, 1.4, 2.5] #seconds
    delays2 = [0.2, 0.4, 1, 2] #seconds
    asr1 = (0.02,0.3,0.1)
    asr2 = (0.1,0.3,0.2)

    n = 1:N
    spectrum1 = @. 1/n
    spectrum2 = @. 0*1/n^4
    spectrum2[1] = 1
    spectrum2[3] = 1/2;
    spectrum2[5] = 1/3;

    x1 = make_source(t, spectrum1, notes1, delays1, asr1)
    x2 = make_source(t, spectrum2, notes2, delays2, asr2)
    return x1 , x2
end

w = 300 # window width
hop = w÷2 - 1 # number of samples to hop over
window = hann(w)
mystft(y) = stft(y, window, hop) #make a single parameter 
myistft(Y) = istft(Y, window, hop)

#Φ = angle.(mystft(y))
#x1_estimate = istft(X1_estimate .* exp.(im .* Φ))

"""
doubles a 3-tensor D in the third dimention
"""
function double_tensor(D)
    D1 = cat(D,0 .* D,dims=3)
    D2 = cat(0 .* D,D,dims=3)
    DD = cat(D1,D2,dims=1)
    return DD
end

"""
main script function for making the spectogram and the fixed factorization tensor DD
"""
function make_spectogram()
    # Generate time points
    sample_rate = 44000. / 16. # 16th reduction from typical sample rate
    t = range(0, 3, step=1/sample_rate)

    # Matrix Sizes
    nq = sample_rate÷2 #Niquist rate
    #tmax = maximum(t)
    F, T = size(mystft(t))
    #r = length(notes)
    freqs = range(0, nq, F)
    #times = range(0, tmax, T)

    L = 52 #88  # of pitches
    N = 6  # of harmonics
    #j = 1:F #j = 1:J
    l = 1:L
    n = 1:N

    h = n #need the alias for Einsum
    f = keynumber_to_frequency.(l) #pitch of the l'th piano note
    ν = freqs #need the alias for Einsum

    # Make large tensor
    @einsum D[l,j,n] := close_in_frequency(h[n]*f[l], ν[j])

    DD = double_tensor(D)

    # make sources
    x1, x2 = make_sources(t, N)
    y = x1 + x2

    # make spectograms
    Y = abs.(mystft(y))
    Φ = angle.(mystft(y))
    Xs = [abs.(mystft(x)) for x ∈ (x1, x2)]

    return Y, Φ, Xs, DD
end

#(A1, A2, b1, b2, error) = als_seperate(Y', DD; maxiter=800, tol=1e-3, λA=0, λb=0, ϵA=0, ϵb=0)
