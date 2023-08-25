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

const PyMod = Module()
const JlMod = Module()

function embed(out)
    isnothing(out) && return ""
    embed_display(out)
end

function convert_py_output(pyout)
    if hasproperty(pyout, :__class__)
        cname = pyconvert(String, pyout.__class__.__name__)
        if cname ∈ ("ndarray", "list")
            pyconvert(Array, pyout)
        elseif cname == "DataFrame"
            DataFrame(PyTable(pyout))
        elseif cname == "dict"
            pyconvert(Dict, pyout)
        else
            pyconvert(Any, pyout)
        end
    else
        pyconvert(Any, pyout)
    end
end
function py_output(lastline)
    if match(r"plt\.show()", lastline) != nothing
        pyeval("plt.gcf()", PyMod)
    elseif match(r"^\w* ?= ?\w", lastline) == nothing &&
           match(r"^ ", lastline) == nothing &&
           match(r"^\t", lastline) == nothing
        pyeval(lastline, PyMod) |> convert_py_output
    else
        nothing
    end
end

_eval(ex) = Base.eval(JlMod, ex)

"""
    mlcode(jlcode, pycode_str; eval = true, showoutput = true, collapse = nothing)

Writes julia and python code in one Pluto cell and shows the language selector switch. Julia code and python code is either a `String` or `nothing`.
If collapse is set to `true` or some title string, the code block will be collapsed with `title = "Details"`, if `collapse = true` and `title = collapse`, if collapse is a string.

## Examples
```
mlcode(
"rand(12)"
,
"np.random.random(12)"
)

mlcode(
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

mlcode(
"2+2"
,
py"2+2"
;
eval = false,
collapse = "click to see more"
)
```
"""
function mlcode(jlcode, pycode; eval = true, showoutput = true, showinput = true, collapse = nothing)
    nojl = jlcode === nothing
    nopy = pycode === nothing
	ojl = if eval && !nojl
        if isa(jlcode, String)
            _eval(Meta.parse("begin\n"*jlcode*"\nend"))
        else
            error("jlcode is a $(typeof(jlcode)) but needs to be a `String`.")
        end
    end
    opy = if eval && !nopy && pycode != ""
        if isa(pycode, String)
            lines = split(pycode, '\n')
            lastline = lines[end-1]
            pyexec(join(lines[1:end-2], "\n"), PyMod)
            py_output(lastline)
        else
            error("pycode is a $(typeof(pycode)) but needs to be a `String`.")
        end
    end
	s1 = nojl ? nothing : @htl """
<pre style="border-radius:0px; padding:5px 20px;">
<code class="language-julia hljs" style="padding:0px 3px; width:650px">
$jlcode
</code>
</pre>
"""
	s2 = nopy ? nothing : @htl """
<pre style="border-radius:0px; padding:5px 20px">
<code class="language-python hljs" style="padding:0px 3px; width:650px">
$pycode
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

