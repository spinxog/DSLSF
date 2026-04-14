# Use pseudo-labels from vetted predictions when available
        pseudo_labels = self.pseudo_labeler.generate_pseudo_labels(
            vetting_results['vetted_predictions'], sequence
        )