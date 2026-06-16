try:
    from cellflow.external._scvi import CFJaxSCVI
except ImportError as e:
    raise ImportError(
        "cellflow.external requires more dependencies. Please install via pip install 'cellflow[external]'"
    ) from e
