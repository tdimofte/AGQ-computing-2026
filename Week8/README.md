### Lecture 8: Scientific computing with Julia (Job Feldbrugge) ### 

Lecture slides and notebooks for this week's workshop. I will introduce the Julia programming language and show how you can use it in High Performance Scientific computing. 

Before the lecture, please install Julia via the terminal following the instructions at: https://julialang.org/downloads/
(Tudor: this may require some troubleshooting, depending on how your system is set up. Putting errors into Claude/GPT/etc. and asking for help can be very useful.)

I recommend running Julia in Visual Studio Code with the Julia Extension: https://code.visualstudio.com/docs/languages/julia

For an overview of the language see the documentation: https://docs.julialang.org/en/v1/

I highly recommend installing the relevant libraries before attending the Workshop section of this lecture. Download the Manifest.toml and Project.toml files into your working directory. Open Julia in the terminal and run 

```import Pkg; Pkg.instantiate()```

Alternatively, once you open a Jupyter notebook, run

```using Pkg```

```Pkg.instantiate()```
