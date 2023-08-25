const NOTEBOOKS = [("Introduction", "introduction.jl"),
                   ("Supervised Learning", "supervised_learning.jl"),
                   ("Generalized Linear Regression", "generalized_linear_regression.jl"),
                   ("Flexibility and Bias-Variance-Decomposition", "flexibility.jl"),
                   ("Model Assessment", "model_evaluation.jl"),
                   ("Regularization", "regularization.jl"),
                   ("Transformations of Input or Output", "transformations.jl"),
                   ("Gradient Descent", "gradient_descent.jl"),
                   ("Multilayer Perceptrons", "mlp.jl"),
                   ("Other Nonlinear Methods", "other_nonlinear.jl"),
#                    ("Recurrent Neural Networks", "rnn.jl"),
#                    ("Tree-Based Methods", "trees.jl"),
                   ("Clustering", "clustering.jl"),
                   ("Principal Component Analysis", "pca.jl"),
                   ("Reinforcement Learning", "rl.jl")
                  ]

function _linkname(path, nb, basedir)
    if haskey(ENV, "html_export") && ENV["html_export"] == "true"
        joinpath(basedir, "$(splitext(nb)[1]).html")
    else
        "open?path=" * joinpath(path, nb)
    end
end
function list_notebooks(file; path = rel_path("notebooks"), basedir = "")
    sp = splitpath(file)
    filename = split(sp[end], "#")[1]
    list = join(["1. " * (nb == filename ?
                            name * " (this notebook)" :
                            "[$name](" * _linkname(path, nb, basedir) * ")")
                 for (name, nb) in NOTEBOOKS], "\n")
Markdown.parse("""# Course Overview

               $list

               Optional Extras:
               1. [Transfer Learning](https://lcnwww.epfl.ch/bio322/transfer_learning.html)
               2. [Generative Models](https://lcnwww.epfl.ch/bio322/ae_gan.html)
               """)
end


function language_selector(id = rand(UInt))
@htl """
<div class="tab">
  <button class = "python_button" id = "python_selector_$id">Python</button>
  <button class = "julia_button" id = "julia_selector_$id">Julia</button>
</div>
<script>
function toggle_code(lang) {
    var myDate = new Date();
    myDate.setMonth(myDate.getMonth() + 12);
	if (lang == "julia") {
		var active_code = document.getElementsByClassName("julia_code")
		var inactive_code = document.getElementsByClassName("python_code")
        var active_button = document.getElementsByClassName("julia_button")
        var inactive_button = document.getElementsByClassName("python_button")
        document.cookie = "mlcourselang=julia";
	} else {
		var inactive_code = document.getElementsByClassName("julia_code")
		var active_code = document.getElementsByClassName("python_code")
        var inactive_button = document.getElementsByClassName("julia_button")
        var active_button = document.getElementsByClassName("python_button")
        document.cookie = "mlcourselang=python";
	}
	for (var i = 0; i < active_code.length; i++) {
    	active_code[i].style.display = "";
  	}
	for (var i = 0; i < inactive_code.length; i++) {
    	inactive_code[i].style.display = "none";
  	}
	for (var i = 0; i < active_button.length; i++) {
        let name = active_button[i].className
		if (name.substring(name.length - 6) != "active") {
			active_button[i].className += " active";
		}
  	}
	for (var i = 0; i < inactive_button.length; i++) {
		inactive_button[i].className = inactive_button[i].className.replace(" active", "");
  	}
}
const python_button_$id = document.getElementById("python_selector_$id")
const julia_button_$id = document.getElementById("julia_selector_$id")
python_button_$id.addEventListener("click", (e) => {
	toggle_code("python")
})
julia_button_$id.addEventListener("click", (e) => {
	toggle_code("julia")
})
function readCookie(name) {
	var nameEQ = name + "=";
	var ca = document.cookie.split(';');
	for(var i=0;i < ca.length;i++) {
		var c = ca[i];
		while (c.charAt(0)==' ') c = c.substring(1,c.length);
		if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length,c.length);
	}
	return "julia";
}
toggle_code(readCookie("mlcourselang"))
</script>

"""
end
const CSS_STYLE =
    @htl """
    <style>
    .tab {
      #overflow: hidden;
      #border: 1px solid #ccc;
      background-color: #f4f4f4;
    }

    .tab button {
      background-color: inherit;
      color: #bbb;
      border-radius: 0px;
      float: left;
      border: none;
      outline: none;
      cursor: pointer;
      padding: 5px 5px;
      transition: 0.3s;
    }

    .tab button:hover {
      background-color: #ded;
      color: #000;
    }

    .tab button.active {
      background-color: #cdc;
      color: #000
    }

    .tabcontent {
      display: none;
      padding: 6px 12px;
      border: 1px solid #ccc;
      border-top: none;
    }

    .collapsible {
      background-color: #AAA;
      color: white;
      cursor: pointer;
      padding: 18px;
      width: 100%;
      border: none;
      text-align: left;
      outline: none;
      font-size: 15px;
    }

    .active, .collapsible:hover {
      background-color: #999;
    }

    .collapsiblecontent {
      padding: 0 18px;
      max-height: 0;
      overflow: hidden;
      transition: max-height 0.2s ease-out;
    }

    </style>


    """

const PYTHON_IMPORTS = String[]
function process_pystr(pystr)
    pystr.args[end] == "" && return pystr
    lines = split(pystr.args[end], "\n")
    lines[end] == "" && pop!(lines)
    to_delete = Int[]
    block_str = join(PYTHON_IMPORTS, "\n") * "\n"
    last_line_extra = true
    for (i, line) in pairs(lines)
        if !isnothing(match(r"import", line))
            push!(to_delete, i)
            if line ∉ PYTHON_IMPORTS
                push!(PYTHON_IMPORTS, string(line))
            end
        elseif i < length(lines)
            block_str *= string(line) * "\n"
        elseif line[1] == ' ' || line[1] == '\t'
            block_str *= string(line)
            last_line_extra = false
        end
    end
    delete_items!(lines, to_delete)
    Expr(:block,
         :(@py_str $block_str),
         last_line_extra ? :(@py_str $(lines[end])) : nothing)
end
function delete_items!(array, to_delete)
	for i in sort(to_delete, rev = true)
		deleteat!(array, i)
	end
end
cleanup!(ex::AbstractString) = nothing
function cleanup!(ex)::Nothing
	to_delete = Int[]
    args = ex.args
	for (i, x) in pairs(args)
		if x == Symbol("@py_str")
			args[i] = replace(args[i+2], "\$" => "")
            push!(to_delete, i+2)
		elseif x isa LineNumberNode
			push!(to_delete, i)
		elseif x isa Expr
            if x.head == :macrocall
                args[i] = x.args[1]
            else
			    cleanup!(x)
            end
		end
	end
    delete_items!(args, to_delete)
end
function embed(out)
    isnothing(out) && return ""
    embed_display(out)
end

# from https://discourse.julialang.org/t/documentation-for-how-macros-handle-keyword-arguments/88009
function reorder_macro_kw_params(exs)
    exs = Any[exs...]
    i = findfirst([(ex isa Expr && ex.head == :parameters) for ex in exs])
    if !isnothing(i)
        extra_kw_def = exs[i].args
        for ex in extra_kw_def
            push!(exs, ex isa Symbol ? Expr(:kw, ex, ex) : ex)
        end
        deleteat!(exs, i)
    end
    return Tuple(exs)
end
function _kwvalue(kws, key, default)
    for kw in kws
        kw.args[1] == key && return kw.args[2]
    end
    return default
end
# _eval(ex) = Core.eval(getproperty(Main, Symbol("workspace#", Main.PlutoRunner.moduleworkspace_count[])), ex)
_eval(ex) = eval(ex)

"""
    @mlcode(jlcode, pycode_str; eval = true, showoutput = true, collapse = nothing)

Macro to write julia and python code in one Pluto cell and show the language selector switch. Julia code can be provided as is or as a string. Python code needs to be provided as a @py_str (see examples).
If collapse is set to `true` or some title string, the code block will be collapsed with `title = "Details"`, if `collapse = true` and `title = collapse`, if collapse is a string.

## Examples
```
@mlcode(
rand(12)
,
py"np.random.random(12)"
)

@mlcode(
\"\"\"
a = rand(12)
scatter(a) # some comments
\"\"\"
,
py\"\"\"
a = np.random.random(12)
plt.scatter(range(12), a)
plt.show()
)

@mlcode(
2+2
,
py"2+2"
;
eval = false,
collapse = "click to see more"
)
```
"""
macro mlcode(exs...)
    exs = reorder_macro_kw_params(exs)
    jlcode = exs[1]
    pycode_str = exs[2]
    kw = length(exs) > 2 ? exs[3:end] : (;)
    nojl = jlcode == :nothing
    nopy = pycode_str == :nothing
    doeval = _kwvalue(kw, :eval, true)
    showoutput = _kwvalue(kw, :showoutput, true)
    showinput = _kwvalue(kw, :showinput, true)
    collapse = _kwvalue(kw, :collapse, nothing)
	ojl = if doeval && !nojl
        if isa(jlcode, String)
            _eval(Meta.parse("begin\n"*jlcode*"\nend"))
        else
            _eval(jlcode)
        end
    end
#     dump(jlcode)
#     eval(pycode_str)
#     return
#     @show isa(pycode_str,String) # pycode_str.args[1] pycode_str.args[end]
    opy = if !nopy && !isa(pycode_str, String)
        pycode_str_run = process_pystr(pycode_str)
    #     dump(pycode_str_run)
    #     return
    #     @show pycode_str_run
        if pycode_str_run.args[end] isa Expr &&
           pycode_str_run.args[end].args[end] == "plt.show()"
            pop!(pycode_str_run.args)
            _eval(pycode_str_run)
            filename = tempname() * ".png"
            py"""plt.savefig($filename)"""
            opy = Markdown.parse("""
    ![](data:img/png; base64, $(open(base64encode, filename)))
    """)
        else
            opy = doeval && pycode_str_run.args[end] != "" ? _eval(pycode_str_run) : nothing
#             if hasproperty(opy, :__class__) && opy.__class__.__name__ == "DataFrame"
#                 names = pystr.(opy.columns.values)
#                 cols = [get(opy, Symbol(name)).values for name in names]
#                 @show names[end] cols[end].__class__.__name__
#                 #opy = DataFrame(cols[end:end], names[end:end])
#             end
            opy
        end
    end
    isa(jlcode, String) || nojl || cleanup!(jlcode)
	s1 = nojl ? nothing : @htl """
<pre style="border-radius:0px; padding:5px 20px;">
<code class="language-julia hljs" style="padding:0px 3px; width:650px">
$(!isa(jlcode, String) && jlcode.head == :block ? join(string.(jlcode.args), "\n") : string(jlcode))
</code>
</pre>
"""
	isa(pycode_str, String) || nopy || cleanup!(pycode_str)
	s2 = nopy ? nothing : @htl """
<pre style="border-radius:0px; padding:5px 20px">
<code class="language-python hljs" style="padding:0px 3px; width:650px">
$(isa(pycode_str, String) ? pycode_str : strip(replace(join(pycode_str.args, ""), "\"" => "", "\\n" => "\n"), '\n'))
</code>
</pre>
"""
result = @htl("""
$(!nopy && !nojl && showinput ? language_selector() : nothing)

<div class="julia_code">
    $(showinput ? s1 : nothing)

    $(showoutput && !nojl ? embed(ojl) : nothing)
</div>
<div class="python_code">
    $(showinput ? s2 : nothing)

    $(showoutput && !nopy ? embed(opy) : nothing)
</div>
""")
if !isnothing(collapse)
    MLCourse.collapse(result, title = isa(collapse, String) ? collapse : "Details")
else
    result
end
end

function collapse(s; title = "Details")
@htl("""
<button class="collapsible">$title</button>
<div class="collapsiblecontent">

$s

</div>
""")
end

# This is a hack
function HypertextLiteral.show(io::IO, ::MIME"text/html", md::Markdown.MD)
    if length(md.content) == 1 && isa(md.content[1], Markdown.Paragraph)
        Markdown.htmlinline(io, md.content[1].content)
    else
        html(io, md)
    end
end
"""
    mlstring(jl_string, py_string)

Returns an html string that shows `jl_string` when julia is selected and `py_string' when python is selected.

## Examples
```
mlstring("This is some explanation of julia code.",
         "This is a totally different explanation of python code.")

@htl "This is some explanation to shows for any language, but adapts to the language, e.g. to first element of an array is indexed by \$(mlstring("1 in julia", "0 in python"))."
```
"""
function mlstring(jstr, pystr)
    @htl """
    <span class="julia_code">
    $jstr
    </span>
    <span class="python_code">
    $pystr
    </span>
    """
end


const FOOTER = @htl """
    <script>
    var coll = document.getElementsByClassName("collapsible");
    var i;

    for (i = 0; i < coll.length; i++) {
      coll[i].addEventListener("click", function() {
        this.classList.toggle("active");
        var content = this.nextElementSibling;
        if (content.style.maxHeight){
          content.style.maxHeight = null;
        } else {
          content.style.maxHeight = content.scrollHeight + "px";
        }
      });
    }
    </script>

<p> This page is part of an <a href="https://bio322.epfl.ch">introductory machine learning course</a> taught by Johanni Brea.<br>The course is inspired by <a href="https://www.statlearning.com/">"An Introduction to Statistical Learning"</a>.</p> <a href="https://www.epfl.ch"><img src="https://www.epfl.ch/wp/5.5/wp-content/themes/wp-theme-2018/assets/svg/epfl-logo.svg"></img></a>
"""

