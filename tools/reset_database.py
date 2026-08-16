import sys
import os


# adiciona a raiz do projeto no caminho do Python
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


from data.database import Database



db = Database()


cursor = db.connection.cursor()


cursor.execute(
    """
    DROP TABLE IF EXISTS indicator_snapshots;
    """
)


db.connection.commit()


print(
    "Tabela indicator_snapshots removida com sucesso."
)


db.close()