class CodegenError(Exception):
    pass


class CyclicDependencyError(CodegenError):
    pass


class UnsupportedEntityError(CodegenError):
    pass
