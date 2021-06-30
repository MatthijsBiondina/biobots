import traceback


class TodoException(Exception):
    def __init__(self):
        super().__init__()
    # def __init__(self, message=None):
    #     if message is None:
    #         trace = traceback.extract_stack()[-2]
    #
    #         filename = trace.filename.split('/')
    #         filename = '/'.join(filename[filename.index('pybrown')+1:])
    #
    #         self.message = f"{trace.name} @ .../{filename}, line {trace.lineno}"
    #     else:
    #         self.message = message
    #     super().__init__(self.message)
