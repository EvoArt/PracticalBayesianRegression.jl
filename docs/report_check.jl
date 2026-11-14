# Runs the report docs exactly as written: the @example blocks of
# howto/documents.md and tutorials/report.md, then writes the Typst, LaTeX
# and Quarto sources those pages show, for CI to compile (report.yml).
#   julia --project=ENV docs/report_check.jl docs/src WORKDIR
using PracticalBayesianRegression

src, work = ARGS[1], ARGS[2]
mkpath(work)

blocks(md, tag) = [m.captures[1] for m in eachmatch(Regex("```" * tag * "\\n(.*?)\\n```", "s"), md)]

function run_examples(file, dir)
    md = read(file, String)
    mod = Module(gensym())
    Core.eval(mod, :(using PracticalBayesianRegression))
    cd(mkpath(dir)) do
        for (k, code) in enumerate(blocks(md, "@example \\w+"))
            code = replace(code, r" *# hide\n?" => "\n")
            println("── ", basename(file), " block ", k)
            v = Base.invokelatest(include_string, mod, code)
            v === nothing || (show(stdout, MIME"text/plain"(), v); println())
        end
    end
    return md
end

doc = run_examples(joinpath(src, "howto", "documents.md"), joinpath(work, "docs"))
write(joinpath(work, "docs", "report", "report.typ"), only(blocks(doc, "typst")))
write(joinpath(work, "docs", "report", "report.tex"), only(blocks(doc, "latex")))

tut = run_examples(joinpath(src, "tutorials", "report.md"), joinpath(work, "tut"))
qmd = only(m.captures[1] for m in eachmatch(r"````markdown\n(.*?)\n````"s, tut))
write(joinpath(work, "tut", "report.qmd"), qmd)
println("WROTE SOURCES")
