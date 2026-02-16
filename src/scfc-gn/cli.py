import typer 


app = typer.Typer(no_args_is_help=True)


@app.command()
def train():
    pass 


@app.command()
def test():
    pass