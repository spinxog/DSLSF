except ImportError as e:
            logging.error(f"Could not import RNA folding pipeline: {e}")
            raise RuntimeError(f"RNA folding pipeline not available: {e}")