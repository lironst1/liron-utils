import docx


# TODO make/edit Word documents

def load_docx(file):
	doc = docx.Document(file)
	return doc
