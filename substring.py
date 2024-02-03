from db import DB


db = DB()

def retrieve_substrings(query):
    cql = """
    SELECT title, body
    FROM colbert.chunks
    """
    rows = db.session.execute(cql)
    L = [{'title': row.title, 'body': row.body} for row in rows if query in row.body]
    return L

if __name__ == '__main__':
    while True:
        query = input('Enter a query: ')
        L = retrieve_substrings(query)
        print(len(L), 'results\n')
        for i, row in enumerate(L):
            print(f"{i+1}. {row['title']}\n{row['body']}")
        print()
