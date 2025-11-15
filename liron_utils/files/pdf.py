def merge_pdf(out, files):
    from PyPDF2 import PdfFileMerger  # pylint: disable=import-error

    h = PdfFileMerger()
    for file in files:
        h.append(file)
    h.write(out)
    h.close()
