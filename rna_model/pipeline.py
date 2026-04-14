def _handle_prediction_error(self, sequence: str, error_msg: str) -> None:
        """Handle prediction errors by logging and re-raising."""
        logging.error(f"Failed to predict structure for sequence {sequence[:20]}...: {error_msg}")
        raise RuntimeError(f"Structure prediction failed: {error_msg}")