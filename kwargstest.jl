function foo(a,b=1;c=1,kwargs...)
    println(a,b,c,kwargs)
    for k ∈ kwargs
        println(k)
    end
end

bar1(a,kwargs...) = foo(a,2;kwargs...)

bar1("a";[c=1])

#=
I want to be able to call foo with a different default b=2,
and pass all other arguments from bar to foo. In particular,
I want to keep foo's default c=1 without explicitly restating it. 
=#