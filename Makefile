data = create_sample_dataset()
sequences = data['sequences'][:5]

# Create test structures for validation
structures = []
for seq in sequences:
    coords = torch.randn(len(seq), 3, 3).numpy()
    from rna_model.data import RNAStructure
    structures.append(RNAStructure(seq, coords, [], [], 'A'))