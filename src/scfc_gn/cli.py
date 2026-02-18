import typer 
from scfc_gn.ucla.load_data import get_all_subjects
from scfc_gn.ucla.models import Subject

app = typer.Typer(no_args_is_help=True)


@app.command()
def train():
    #load data

    #get loader 

    #train loop 
    pass 


@app.command()
def test():
    pass


@app.command()
def get_subjects():
    subs, group_sub = get_all_subjects()
    print(len(subs))

