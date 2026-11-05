@testset "formula parsing" begin
    f = @bf(y ~ x * z + log(w) + (1 + x | g) + (1 || g:h))
    rhs = f.dpars[1].rhs
    @test rhs.intercept
    @test [join((a.label for a in t), ":") for t in rhs.terms] == ["x", "z", "log(w)", "x:z"]
    @test length(rhs.groups) == 2
    @test rhs.groups[1].cor && !rhs.groups[2].cor
    @test PG.group_label(rhs.groups[2]) == "g:h"

    # `(1 | g:h)` parses as `(1 | g):h` in Julia; must still be a group term
    f2 = bf("y ~ x + (1 | g:h)")
    @test length(f2.dpars[1].rhs.groups) == 1
    @test PG.group_label(f2.dpars[1].rhs.groups[1]) == "g:h"

    # nesting expands like lme4/brms
    f3 = @bf(y ~ (1 | a / b))
    @test [PG.group_label(g) for g in f3.dpars[1].rhs.groups] == ["a", "a:b"]

    # intercept removal
    @test !(@bf(y ~ 0 + x)).dpars[1].rhs.intercept
    @test !(@bf(y ~ x - 1)).dpars[1].rhs.intercept

    # addition terms and distributional formulas
    f4 = @bf(y | trials(n) + weights(w) ~ x, sigma ~ z)
    @test f4.trials isa PG.Atom && f4.weights isa PG.Atom
    @test [d.dpar for d in f4.dpars] == [:mu, :sigma]
    @test_throws ArgumentError bf("y | notaterm(z) ~ x")

    # R precedence for `:` although Julia binds it more loosely than `+`
    lab(f) = [join((a.label for a in t), ":") for t in f.dpars[1].rhs.terms]
    @test lab(@bf(y ~ x + a:b)) == ["x", "a:b"]
    @test lab(@bf(y ~ x + a:b + c)) == ["x", "c", "a:b"]
    @test lab(bf("y ~ x + a:b:c + z")) == ["x", "z", "a:b:c"]
    @test lab(@bf(y ~ a * b:c)) == ["a", "b:c", "a:b:c"]
    @test lab(@bf(y ~ x + a & b)) == ["x", "a:b"]
    @test PG.group_label((@bf(y ~ x + (1 + x | g:h))).dpars[1].rhs.groups[1]) == "g:h"

    # string and macro forms agree
    @test bf("y ~ x + (1|g)").text == (@bf(y ~ x + (1 | g))).text
    # brms's lf(): + adds distributional formulas
    f1 = @bf(y ~ x) + @lf(sigma ~ z, nu ~ 1)
    f2 = @bf(y ~ x, sigma ~ z, nu ~ 1)
    @test [d.dpar for d in f1.dpars] == [d.dpar for d in f2.dpars] && f1.text == f2.text
    @test (bf("y ~ x") + lf("sigma ~ z")).dpars[2].dpar === :sigma
    @test_throws ArgumentError @bf(y ~ x, sigma ~ 1) + @lf(sigma ~ z)
end
