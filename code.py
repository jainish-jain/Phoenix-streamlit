import re
import streamlit as st
from streamlit.hashing import Context
from streamlit_ace import st_ace
import sys
from io import StringIO
import contextlib
from streamlit import cli as stcli
import base64
LANGUAGES = [
    "abap", "abc", "actionscript", "ada", "alda", "apache_conf", "apex", "applescript", "aql", 
    "asciidoc", "asl", "assembly_x86", "autohotkey", "batchfile", "c9search", "c_cpp", "cirru", 
    "clojure", "cobol", "coffee", "coldfusion", "crystal", "csharp", "csound_document", "csound_orchestra", 
    "csound_score", "csp", "css", "curly", "d", "dart", "diff", "django", "dockerfile", "dot", "drools", 
    "edifact", "eiffel", "ejs", "elixir", "elm", "erlang", "forth", "fortran", "fsharp", "fsl", "ftl", 
    "gcode", "gherkin", "gitignore", "glsl", "gobstones", "golang", "graphqlschema", "groovy", "haml", 
    "handlebars", "haskell", "haskell_cabal", "haxe", "hjson", "html", "html_elixir", "html_ruby", "ini", 
    "io", "jack", "jade", "java", "javascript", "json", "json5", "jsoniq", "jsp", "jssm", "jsx", "julia", 
    "kotlin", "latex", "less", "liquid", "lisp", "livescript", "logiql", "logtalk", "lsl", "lua", "luapage", 
    "lucene", "makefile", "markdown", "mask", "matlab", "maze", "mediawiki", "mel", "mixal", "mushcode", 
    "mysql", "nginx", "nim", "nix", "nsis", "nunjucks", "objectivec", "ocaml", "pascal", "perl", "perl6", 
    "pgsql", "php", "php_laravel_blade", "pig", "plain_text", "powershell", "praat", "prisma", "prolog", 
    "properties", "protobuf", "puppet", "python", "qml", "r", "razor", "rdoc", "red", "redshift", "rhtml", 
    "rst", "ruby", "rust", "sass", "scad", "scala", "scheme", "scss", "sh", "sjs", "slim", "smarty", 
    "snippets", "soy_template", "space", "sparql", "sql", "sqlserver", "stylus", "svg", "swift", "tcl", 
    "terraform", "tex", "text", "textile", "toml", "tsx", "turtle", "twig", "typescript", "vala", "vbscript", 
    "velocity", "verilog", "vhdl", "visualforce", "wollok", "xml", "xquery", "yaml"
]
THEMES = [
    "ambiance", "chaos", "chrome", "clouds", "clouds_midnight", "cobalt", "crimson_editor", "dawn",
    "dracula", "dreamweaver", "eclipse", "github", "gob", "gruvbox", "idle_fingers", "iplastic",
    "katzenmilch", "kr_theme", "kuroir", "merbivore", "merbivore_soft", "mono_industrial", "monokai",
    "nord_dark", "pastel_on_dark", "solarized_dark", "solarized_light", "sqlserver", "terminal",
    "textmate", "tomorrow", "tomorrow_night", "tomorrow_night_blue", "tomorrow_night_bright",
    "tomorrow_night_eighties", "twilight", "vibrant_ink", "xcode"
]

KEYBINDINGS = [
    "emacs", "sublime", "vim", "vscode"
]

@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

def btndf(btnname,link):
    custom_css = f""" 
        <style>
            #button_id{{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #button_id:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #button_id:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a id="button_id" href={link} target="_blank">{btnname}</a><br></br>'
    return dl_link


if __name__ == "__main__":

    st.set_page_config(page_title="Phoenix" ,page_icon="favicon.png")

    #from app import btndf
    st.sidebar.markdown(btndf("For phoenix's Workspace",'https://share.streamlit.io/jainish-jain/phoenix/main/app.py'),unsafe_allow_html=True)
    #st.title("Phoenix's Program Workspace")
    LOGO_IMAGE = "favicon.png"

    st.markdown(
        """
        <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Belleza" />
        <style>
        .container {
            display: flex;
        }
        .logo-text {
            font-family: "Belleza", sans-serif;
            # font-weight:10 !important;
            font-size:40px !important;
            #color: #f9a01b !important;
            padding-top: 5px !important;
            padding-left: 5px !important;
        }
        .logo-img {
            float:right;
            width:90px;
            height:80px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
            <p class="logo-text">Phoenix Program Workspace</p>
        </div>
        """,
        unsafe_allow_html=True
    )



    st.subheader(":memo: Python ")
    st.text("\n")
    content = st_ace(
        placeholder='Enter Your Code Here...',
        language="python",
        theme=st.sidebar.selectbox("Theme", options=THEMES, index=11),
        keybinding=st.sidebar.selectbox("Keybinding mode", options=KEYBINDINGS, index=3),
        font_size=st.sidebar.slider("Font size", 5, 24, 16),
        tab_size=st.sidebar.slider("Tab size", 1, 8, 4),
        show_gutter=st.sidebar.checkbox("Show gutter", value=True),
        show_print_margin=st.sidebar.checkbox("Show print margin", value=True),
        wrap=st.sidebar.checkbox("Wrap enabled", value=True),
        auto_update=True,
        readonly=st.sidebar.checkbox("Read-only", value=False, key="ace-editor"),
        key="ace-editor"
    )
   
        #sys.base_exec_prefix('90')
    
    # with stdoutIO() as s:
    #     try:
    #         #sys.exec_prefix(98)
    #         exec(content)
    #         sys.base_exec_prefix('90')
    #     except:
    #         e=sys.exc_info()    
    #         st.error(e)
    if st.button("Run"):
        try:
            with stdoutIO() as s:
                exec(content)
            pr=s.getvalue().split('\n')
        except:
            e=sys.exc_info()
            pr=None   
        st.subheader('Output')
       
        #print(s.getvalue())
        if pr!=None:
            for i in pr:
                st.write(i)
        else:   
            st.error(e)
    #st.markdown("""<iframe src="https://trinket.io/embed/python/db5ab779e2?toggleCode=true&start=result&runMode=console" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe> """,unsafe_allow_html=True)
    # sys.argv = [88, 90]
    # execfile(exec(content))
    # print('q') 