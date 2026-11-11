import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from main_model import main_model
import scanpy as sc
import episcanpy as epi


if __name__ == "__main__":
    datasets = ['SNARE']
    for dataset in datasets:
        if dataset == 'SNARE':
            train_rna_data_path = 'data/SNARE/RNA.h5ad'
            train_atac_data_path = 'data/SNARE/ATAC.h5ad'
            annData_rna = sc.read_h5ad(train_rna_data_path)
            annData_atac = sc.read_h5ad(train_atac_data_path)
            epi.pp.select_var_feature(annData_atac, nb_features=4000, show=False, copy=False)
            sc.pp.highly_variable_genes(annData_rna, n_top_genes=4000, flavor="seurat_v3", subset=True)
            annData_rna.obs['label'] = annData_rna.obs['cell_type']
            annData_atac.obs['label'] = annData_atac.obs['cell_type']
            main_model(annData_rna, annData_atac, dataset)

        if dataset == '10X10k_CITE':
            train_rna_data_path = 'data/10X10k_CITE/RNA.h5ad'
            train_adt_data_path = 'data/10X10k_CITE/ADT.h5ad'
            annData_rna = sc.read_h5ad(train_rna_data_path)
            annData_adt = sc.read_h5ad(train_adt_data_path)
            common_cells = annData_rna.obs_names.intersection(annData_adt.obs_names)
            annData_rna = annData_rna[common_cells].copy()
            annData_adt = annData_adt[common_cells].copy()
            if 'cell_type' in annData_rna.obs.columns:
                mask = ~annData_rna.obs['cell_type'].str.contains('doublets', case=False, na=False)
                annData_rna = annData_rna[mask].copy()
                annData_adt = annData_adt[mask].copy()
            sc.pp.highly_variable_genes(annData_rna, n_top_genes=4000, flavor="seurat_v3", subset=True)
            annData_rna.obs['label'] = annData_rna.obs['cell_type']
            annData_adt.obs['label'] = annData_adt.obs['cell_type']
            main_model(annData_rna, annData_adt, dataset)

        if dataset == 'GSE128639_CITE':
            train_rna_data_path = 'data/GSE128639_CITE/RNA.h5ad'
            train_adt_data_path = 'data/GSE128639_CITE/ADT.h5ad'
            annData_rna = sc.read_h5ad(train_rna_data_path)
            annData_adt = sc.read_h5ad(train_adt_data_path)
            common_cells = annData_rna.obs_names.intersection(annData_adt.obs_names)
            annData_rna = annData_rna[common_cells].copy()
            annData_adt = annData_adt[common_cells].copy()
            if 'cell_type' in annData_rna.obs.columns:
                mask = ~annData_rna.obs['cell_type'].str.contains('doublets', case=False, na=False)
                annData_rna = annData_rna[mask].copy()
                annData_adt = annData_adt[mask].copy()
            sc.pp.highly_variable_genes(annData_rna, n_top_genes=4000, flavor="seurat_v3", subset=True)
            annData_rna.obs['label'] = annData_rna.obs['cell_type']
            annData_adt.obs['label'] = annData_adt.obs['cell_type']
            main_model(annData_rna, annData_adt, dataset)

        if dataset == 'spleen_lymph_CITE_206':
            train_rna_data_path = 'data/spleen_lymph_CITE_206/RNA.h5ad'
            train_adt_data_path = 'data/spleen_lymph_CITE_206/ADT.h5ad'
            annData_rna = sc.read_h5ad(train_rna_data_path)
            annData_adt = sc.read_h5ad(train_adt_data_path)
            common_cells = annData_rna.obs_names.intersection(annData_adt.obs_names)
            annData_rna = annData_rna[common_cells].copy()
            annData_adt = annData_adt[common_cells].copy()
            if 'cell_type' in annData_rna.obs.columns:
                mask = ~annData_rna.obs['cell_type'].str.contains('doublets', case=False, na=False)
                annData_rna = annData_rna[mask].copy()
                annData_adt = annData_adt[mask].copy()
            sc.pp.highly_variable_genes(annData_rna, n_top_genes=4000, flavor="seurat_v3", subset=True)
            annData_rna.obs['label'] = annData_rna.obs['cell_type']
            annData_adt.obs['label'] = annData_adt.obs['cell_type']
            main_model(annData_rna, annData_adt, dataset)

        if dataset == 'spleen_lymph_CITE_111':
            train_rna_data_path = 'data/spleen_lymph_CITE_111/RNA.h5ad'
            train_adt_data_path = 'data/spleen_lymph_CITE_111/ADT.h5ad'
            annData_rna = sc.read_h5ad(train_rna_data_path)
            annData_adt = sc.read_h5ad(train_adt_data_path)
            common_cells = annData_rna.obs_names.intersection(annData_adt.obs_names)
            annData_rna = annData_rna[common_cells].copy()
            annData_adt = annData_adt[common_cells].copy()
            if 'cell_type' in annData_rna.obs.columns:
                mask = ~annData_rna.obs['cell_type'].str.contains('doublets', case=False, na=False)
                annData_rna = annData_rna[mask].copy()
                annData_adt = annData_adt[mask].copy()
            sc.pp.highly_variable_genes(annData_rna, n_top_genes=4000, flavor="seurat_v3", subset=True)
            annData_rna.obs['label'] = annData_rna.obs['cell_type']
            annData_adt.obs['label'] = annData_adt.obs['cell_type']
            main_model(annData_rna, annData_adt, dataset)

        if dataset == 'PBMC':
            train_rna_data_path = 'data/PBMC/RNA.h5ad'
            train_atac_data_path = 'data/PBMC/ATAC.h5ad'
            annData_rna = sc.read_h5ad(train_rna_data_path)
            annData_atac = sc.read_h5ad(train_atac_data_path)
            epi.pp.select_var_feature(annData_atac, nb_features=4000, show=False, copy=False)
            sc.pp.highly_variable_genes(annData_rna, n_top_genes=4000, flavor="seurat_v3", subset=True)
            annData_rna.obs['label'] = annData_rna.obs['cell_type']
            annData_atac.obs['label'] = annData_atac.obs['cell_type']
            main_model(annData_rna, annData_atac, dataset)

        if dataset == 'NeurIPS_BMMC':
            train_rna_data_path = 'data/NeurIPS_BMMC/RNA.h5ad'
            train_atac_data_path = 'data/NeurIPS_BMMC/ATAC.h5ad'
            annData_rna = sc.read_h5ad(train_rna_data_path)
            annData_atac = sc.read_h5ad(train_atac_data_path)
            epi.pp.select_var_feature(annData_atac, nb_features=4000, show=False, copy=False)
            sc.pp.highly_variable_genes(annData_rna, n_top_genes=4000, flavor="seurat_v3", subset=True)
            annData_rna.obs['label'] = annData_rna.obs['cell_type']
            annData_atac.obs['label'] = annData_atac.obs['cell_type']
            main_model(annData_rna, annData_atac, dataset)

        if dataset == 'SHARE':
            train_rna_data_path = 'data/SHARE/RNA.h5ad'
            train_atac_data_path = 'data/SHARE/ATAC.h5ad'
            annData_rna = sc.read_h5ad(train_rna_data_path)
            annData_atac = sc.read_h5ad(train_atac_data_path)
            epi.pp.select_var_feature(annData_atac, nb_features=4000, show=False, copy=False)
            sc.pp.highly_variable_genes(annData_rna, n_top_genes=4000, flavor="seurat_v3", subset=True)
            annData_rna.obs['label'] = annData_rna.obs['cell_type']
            annData_atac.obs['label'] = annData_atac.obs['cell_type']
            main_model(annData_rna, annData_atac, dataset)

        if dataset == 'CITE':
            batch = False
            train_rna_data_path = 'data/CITE/RNA.h5ad'
            train_adt_data_path = 'data/CITE/ADT.h5ad'
            annData_rna = sc.read_h5ad(train_rna_data_path)
            annData_adt = sc.read_h5ad(train_adt_data_path)
            doublet_bool = (annData_rna.obs['celltype.l2'] != 'Doublet')
            annData_rna = annData_rna[doublet_bool].copy()
            annData_adt = annData_adt[doublet_bool].copy()
            donors = ['P2']
            index = [x in donors for x in annData_rna.obs['donor']]
            annData_rna = annData_rna[index].copy()
            annData_adt = annData_adt[index].copy()
            sc.pp.highly_variable_genes(annData_rna, n_top_genes=4000, flavor="seurat_v3", subset=True)
            annData_rna.obs['label'] = annData_rna.obs['celltype.l2']
            annData_adt.obs['label'] = annData_adt.obs['celltype.l2']
            main_model(annData_rna, annData_adt, dataset)
