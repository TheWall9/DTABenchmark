import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json

from toolbox.featurizer.protein import (ProtPssmFeaturizer, ContactMapFeaturizer, PocketGVPFeaturizer, LLMFeaturizer,
                                        SurfaceNormalFeaturizer, PocketSurfaceNormalFeaturizer,
                                        PocketUnimolFeaturizer, PocketSurfaceNormalUnimolFeaturizer,
                                        LLMStructFeaturizer)



if __name__=="__main__":
    map_file = "../data/uniprot_alphafold_struct/seq_struct_esmfold_map.json"
    hhsuite_db_path = "/mnt/data/datasets/Uniclust/uniclust30_2018_08/uniclust30_2018_08"
    with open(map_file) as f:
        struct_seq_map = json.load(f)

    # featurizer = ProtPssmFeaturizer(hhsuite_db_path)
    # featurizer.verbose = True
    # featurizer.featurize(list(struct_seq_map.keys()), verbose=True)
    #
    struct_root_dir = "../data/uniprot_alphafold_struct"
    # featurizer = ContactMapFeaturizer(struct_root_dir)
    # featurizer.featurize(list(struct_seq_map.keys()), verbose=True)

    featurizer = PocketGVPFeaturizer(pretrained_model_name_or_path='facebook/esm2_t6_8M_UR50D', n_res_expand=10, pocket_top=3, pocket_type='dogsite3')
    featurizer.featurize(
        ["MAQKENSYPWPYGRQTAPSGLSTLPQRVLRKEPVTPSALVLMSRSNVQPTAAPGQKVMENSSGTPDILTRHFTIDDFEIGRPLGKGKFGNVYLAREKKSHFIVALKVLFKSQIEKEGVEHQLRREIEIQAHLHHPNILRLYNYFYDRRRIYLILEYAPRGELYKELQKSCTFDEQRTATIMEELADALMYCHGKKVIHRDIKPENLLLGLKGELKIADFGWSVHAPSLRRKTMCGTLDYLPPEMIEGRMHNEKVDLWCIGVLCYELLVGNPPFESASHNETYRRIVKVDLKFPASVPMGAQDLISKLLRHNPSERLPLAQVSAHPWVRANSRRVLPPSALQSVA",
         "MATITCTRFTEEYQLFEELGKGAFSVVRRCVKVLAGQEYAAKIINTKKLSARDHQKLEREARICRLLKHPNIVRLHDSISEEGHHYLIFDLVTGGELFEDIVAREYYSEADASHCIQQILEAVLHCHQMGVVHRDLKPENLLLASKLKGAAVKLADFGLAIEVEGEQQAWFGFAGTPGYLSPEVLRKDPYGKPVDLWACGVILYILLVGYPPFWDEDQHRLYQQIKAGAYDFPSPEWDTVTPEAKDLINKMLTINPSKRITAAEALKHPWISHRSTVASCMHRQETVDCLKKFNARRKLKGAILTTMLATRNFSGGKSGGNKKSDGVKESSESTNTTIEDEDTKVRKQEIIKVTEQLIEAISNGDFESYTKMCDPGMTAFEPEALGNLVEGLDFHRFYFENLWSRNSKPVHTTILNPHIHLMGDESACIAYIRITQYLDAGGIPRTAQSEETRVWHRRDGKWQIVHFHRSGAPSVLPH",
         "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL",
         ], verbose=True,)
    exit()
    featurizer = LLMStructFeaturizer(struct_root_dir, pocket_type='dogsite3', pocket_tops=3,
                                     pretrained_model_name_or_path='facebook/esm2_t6_8M_UR50D')
    featurizer.featurize(list(struct_seq_map.keys()), verbose=True)
    exit()
    featurizer = LLMFeaturizer(pretrained_model_name_or_path='facebook/esm1b_t33_650M_UR50S')
    featurizer.featurize(list(struct_seq_map.keys()), verbose=True)
    # exit()
    for pretrained_model_name_or_path in ['facebook/esm2_t6_8M_UR50D', 'facebook/esm2_t12_35M_UR50D',
                                          'facebook/esm2_t30_150M_UR50D', #'facebook/esm1b_t33_650M_UR50S',
                                          'Rostlab/prot_bert']:
        featurizer = LLMFeaturizer(pretrained_model_name_or_path=pretrained_model_name_or_path)
        featurizer.featurize(list(struct_seq_map.keys()), verbose=True)
        featurizer = LLMStructFeaturizer(struct_root_dir, pocket_type='fpocket', pocket_tops=2,
                                         pretrained_model_name_or_path=pretrained_model_name_or_path)
        featurizer.featurize(list(struct_seq_map.keys()), verbose=True)
    exit()
    featurizer = PocketSurfaceNormalFeaturizer(struct_root_dir, pocket_top=2)
    featurizer.featurize(list(struct_seq_map.keys()), verbose=True)

    featurizer = PocketSurfaceNormalFeaturizer(struct_root_dir, pocket_top=2, pocket_point_nums=256)
    featurizer.featurize(list(struct_seq_map.keys()), verbose=True)

    featurizer = PocketSurfaceNormalFeaturizer(struct_root_dir, pocket_top=2, pocket_point_nums=128)
    featurizer.featurize(list(struct_seq_map.keys()), verbose=True)
    # exit()

    for pretrained_model_name_or_path in ['facebook/esm2_t6_8M_UR50D', ]:
        for pocket_type in ['fpocket', 'dogsite3']:
            for pocket_top in [2,5,10]:
                featurizer = LLMStructFeaturizer(struct_root_dir, pocket_type=pocket_type, pocket_tops=pocket_top,
                                                 pretrained_model_name_or_path=pretrained_model_name_or_path)
                featurizer.featurize(list(struct_seq_map.keys()), verbose=True)
    exit()
    featurizer = PocketGVPFeaturizer(struct_root_dir)
    featurizer.featurize(list(struct_seq_map.keys()), verbose=True)

    featurizer = LLMFeaturizer("facebook/esm2_t6_8M_UR50D")
    featurizer.featurize(list(struct_seq_map.keys()), verbose=True)

    featurizer = LLMFeaturizer("Rostlab/prot_bert")
    featurizer.featurize(list(struct_seq_map.keys()), verbose=True)

    featurizer = SurfaceNormalFeaturizer(struct_root_dir, struct_type='alphafold')
    featurizer.featurize(list(struct_seq_map.keys()), verbose=True)

    featurizer = SurfaceNormalFeaturizer(struct_root_dir, struct_type='esmfold')
    featurizer.featurize(list(struct_seq_map.keys()), verbose=True)

    featurizer = PocketSurfaceNormalFeaturizer(struct_root_dir)
    featurizer.featurize(list(struct_seq_map.keys()), verbose=True)

    featurizer = PocketSurfaceNormalUnimolFeaturizer(struct_root_dir)
    featurizer.featurize(list(struct_seq_map.keys()), verbose=True)




