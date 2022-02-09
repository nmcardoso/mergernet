Search.setIndex({docnames:["api/mergernet.core","api/mergernet.core.artifacts","api/mergernet.core.artifacts.ArtifactHelper","api/mergernet.core.constants","api/mergernet.core.dataset","api/mergernet.core.dataset.Dataset","api/mergernet.core.dataset.DatasetConfig","api/mergernet.core.dataset.DistributionKFold","api/mergernet.core.dataset.StratifiedDistributionKFold","api/mergernet.core.jobs","api/mergernet.core.jobs.BaseJob","api/mergernet.core.jobs.JobRunner","api/mergernet.core.logger","api/mergernet.core.logger.Logger","api/mergernet.core.rgb","api/mergernet.core.rgb.RGB","api/mergernet.core.trilogy","api/mergernet.core.trilogy.MakeImg","api/mergernet.core.trilogy.RGB2im","api/mergernet.core.trilogy.adjust_saturation","api/mergernet.core.trilogy.da","api/mergernet.core.trilogy.get_clip","api/mergernet.core.trilogy.get_levels","api/mergernet.core.trilogy.imscale","api/mergernet.core.trilogy.meanstd","api/mergernet.core.trilogy.rms","api/mergernet.core.trilogy.satK2m","api/mergernet.core.trilogy.setLevel","api/mergernet.core.utils","api/mergernet.core.utils.SingletonMeta","api/mergernet.core.utils.Timming","api/mergernet.core.utils.array_fallback","api/mergernet.core.utils.load_image","api/mergernet.core.utils.load_table","api/mergernet.core.utils.save_table","api/mergernet.jobs","api/mergernet.jobs.j001_test_train","api/mergernet.jobs.j001_test_train.Job","api/mergernet.jobs.j002_hyperparam","api/mergernet.jobs.j002_hyperparam.Job","api/mergernet.model","api/mergernet.model.baseline","api/mergernet.model.baseline.ConvolutionalClassifier","api/mergernet.model.baseline.Metamodel","api/mergernet.model.callback","api/mergernet.model.callback.DeltaStopping","api/mergernet.model.hypermodel","api/mergernet.model.hypermodel.BayesianTuner","api/mergernet.model.hypermodel.SimpleHyperModel","api/mergernet.model.plot","api/mergernet.model.plot.Serie","api/mergernet.model.plot.color_color","api/mergernet.model.plot.data_distribution","api/mergernet.model.plot.roc","api/mergernet.model.plot.train_metrics","api/mergernet.model.preprocessing","api/mergernet.model.preprocessing.load_jpg","api/mergernet.model.preprocessing.normalize_rgb","api/mergernet.model.preprocessing.one_hot","api/mergernet.model.preprocessing.standardize_rgb","api/mergernet.services","api/mergernet.services.github","api/mergernet.services.github.GithubService","api/mergernet.services.google","api/mergernet.services.google.GDrive","api/mergernet.services.imaging","api/mergernet.services.imaging.BaseImagingService","api/mergernet.services.legacy","api/mergernet.services.legacy.LegacyService","api/mergernet.services.sdss","api/mergernet.services.sdss.SloanService","api/mergernet.services.splus","api/mergernet.services.splus.ImageType","api/mergernet.services.splus.SplusService","api/mergernet.services.splus.update_authorization","api/mergernet.services.tensorboard","api/mergernet.services.tensorboard.TensorboardService","api/mergernet.services.utils","api/mergernet.services.utils.append_query_params","api/mergernet.services.utils.batch_download_file","api/mergernet.services.utils.download_file","artifacts_template","experiments","index","jobs/000","reference"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api/mergernet.core.rst","api/mergernet.core.artifacts.rst","api/mergernet.core.artifacts.ArtifactHelper.rst","api/mergernet.core.constants.rst","api/mergernet.core.dataset.rst","api/mergernet.core.dataset.Dataset.rst","api/mergernet.core.dataset.DatasetConfig.rst","api/mergernet.core.dataset.DistributionKFold.rst","api/mergernet.core.dataset.StratifiedDistributionKFold.rst","api/mergernet.core.jobs.rst","api/mergernet.core.jobs.BaseJob.rst","api/mergernet.core.jobs.JobRunner.rst","api/mergernet.core.logger.rst","api/mergernet.core.logger.Logger.rst","api/mergernet.core.rgb.rst","api/mergernet.core.rgb.RGB.rst","api/mergernet.core.trilogy.rst","api/mergernet.core.trilogy.MakeImg.rst","api/mergernet.core.trilogy.RGB2im.rst","api/mergernet.core.trilogy.adjust_saturation.rst","api/mergernet.core.trilogy.da.rst","api/mergernet.core.trilogy.get_clip.rst","api/mergernet.core.trilogy.get_levels.rst","api/mergernet.core.trilogy.imscale.rst","api/mergernet.core.trilogy.meanstd.rst","api/mergernet.core.trilogy.rms.rst","api/mergernet.core.trilogy.satK2m.rst","api/mergernet.core.trilogy.setLevel.rst","api/mergernet.core.utils.rst","api/mergernet.core.utils.SingletonMeta.rst","api/mergernet.core.utils.Timming.rst","api/mergernet.core.utils.array_fallback.rst","api/mergernet.core.utils.load_image.rst","api/mergernet.core.utils.load_table.rst","api/mergernet.core.utils.save_table.rst","api/mergernet.jobs.rst","api/mergernet.jobs.j001_test_train.rst","api/mergernet.jobs.j001_test_train.Job.rst","api/mergernet.jobs.j002_hyperparam.rst","api/mergernet.jobs.j002_hyperparam.Job.rst","api/mergernet.model.rst","api/mergernet.model.baseline.rst","api/mergernet.model.baseline.ConvolutionalClassifier.rst","api/mergernet.model.baseline.Metamodel.rst","api/mergernet.model.callback.rst","api/mergernet.model.callback.DeltaStopping.rst","api/mergernet.model.hypermodel.rst","api/mergernet.model.hypermodel.BayesianTuner.rst","api/mergernet.model.hypermodel.SimpleHyperModel.rst","api/mergernet.model.plot.rst","api/mergernet.model.plot.Serie.rst","api/mergernet.model.plot.color_color.rst","api/mergernet.model.plot.data_distribution.rst","api/mergernet.model.plot.roc.rst","api/mergernet.model.plot.train_metrics.rst","api/mergernet.model.preprocessing.rst","api/mergernet.model.preprocessing.load_jpg.rst","api/mergernet.model.preprocessing.normalize_rgb.rst","api/mergernet.model.preprocessing.one_hot.rst","api/mergernet.model.preprocessing.standardize_rgb.rst","api/mergernet.services.rst","api/mergernet.services.github.rst","api/mergernet.services.github.GithubService.rst","api/mergernet.services.google.rst","api/mergernet.services.google.GDrive.rst","api/mergernet.services.imaging.rst","api/mergernet.services.imaging.BaseImagingService.rst","api/mergernet.services.legacy.rst","api/mergernet.services.legacy.LegacyService.rst","api/mergernet.services.sdss.rst","api/mergernet.services.sdss.SloanService.rst","api/mergernet.services.splus.rst","api/mergernet.services.splus.ImageType.rst","api/mergernet.services.splus.SplusService.rst","api/mergernet.services.splus.update_authorization.rst","api/mergernet.services.tensorboard.rst","api/mergernet.services.tensorboard.TensorboardService.rst","api/mergernet.services.utils.rst","api/mergernet.services.utils.append_query_params.rst","api/mergernet.services.utils.batch_download_file.rst","api/mergernet.services.utils.download_file.rst","artifacts_template.rst","experiments.rst","index.rst","jobs/000.rst","reference.rst"],objects:{"mergernet.core":[[1,0,0,"-","artifacts"],[3,0,0,"-","constants"],[4,0,0,"-","dataset"],[9,0,0,"-","jobs"],[12,0,0,"-","logger"],[14,0,0,"-","rgb"],[16,0,0,"-","trilogy"],[28,0,0,"-","utils"]],"mergernet.core.artifacts":[[2,1,1,"","ArtifactHelper"]],"mergernet.core.artifacts.ArtifactHelper":[[2,2,1,"","_lock"],[2,3,1,"","_upload_gdrive"],[2,3,1,"","_upload_github"],[2,2,1,"","artifact_path"],[2,3,1,"","config"],[2,2,1,"","gdrive_path"],[2,3,1,"","save_json"],[2,3,1,"","upload"],[2,3,1,"","upload_dir"],[2,3,1,"","upload_json"],[2,3,1,"","upload_log"],[2,3,1,"","upload_model"],[2,3,1,"","upload_text"],[2,2,1,"","use_gdrive"],[2,2,1,"","use_github"]],"mergernet.core.dataset":[[5,1,1,"","Dataset"],[6,1,1,"","DatasetConfig"],[7,1,1,"","DistributionKFold"],[8,1,1,"","StratifiedDistributionKFold"]],"mergernet.core.dataset.Dataset":[[5,2,1,"","RGB_CONFIG"],[5,3,1,"","_detect_img_extension"],[5,3,1,"","_discretize_label"],[5,3,1,"","compute_class_weight"],[5,3,1,"","concat_fold_column"],[5,3,1,"","download"],[5,3,1,"","get_fold"],[5,3,1,"","is_dataset_downloaded"]],"mergernet.core.dataset.DistributionKFold":[[7,3,1,"","split"]],"mergernet.core.dataset.StratifiedDistributionKFold":[[8,3,1,"","compute_max_bins"],[8,3,1,"","split_all"],[8,3,1,"","split_ids"]],"mergernet.core.jobs":[[10,1,1,"","BaseJob"],[11,1,1,"","JobRunner"]],"mergernet.core.jobs.BaseJob":[[10,3,1,"","get_system_resources"],[10,3,1,"","post_run"],[10,3,1,"","pre_run"],[10,3,1,"","run"],[10,3,1,"","start_execution"]],"mergernet.core.jobs.JobRunner":[[11,3,1,"","fetch"],[11,3,1,"","list_jobs"],[11,3,1,"","run_job"]],"mergernet.core.logger":[[13,1,1,"","Logger"]],"mergernet.core.logger.Logger":[[13,2,1,"","_lock"],[13,3,1,"","get_logger"]],"mergernet.core.rgb":[[15,1,1,"","RGB"]],"mergernet.core.rgb.RGB":[[15,3,1,"","make_trilogy_fits"],[15,3,1,"","trilogy_fits_to_png"]],"mergernet.core.trilogy":[[17,1,1,"","MakeImg"],[18,4,1,"","RGB2im"],[19,4,1,"","adjust_saturation"],[20,4,1,"","da"],[21,4,1,"","get_clip"],[22,4,1,"","get_levels"],[23,4,1,"","imscale"],[24,4,1,"","meanstd"],[25,4,1,"","rms"],[26,4,1,"","satK2m"],[27,4,1,"","setLevel"]],"mergernet.core.trilogy.MakeImg":[[17,3,1,"","color"],[17,3,1,"","get_array"],[17,3,1,"","savefig"]],"mergernet.core.utils":[[29,1,1,"","SingletonMeta"],[30,1,1,"","Timming"],[31,4,1,"","array_fallback"],[32,4,1,"","load_image"],[33,4,1,"","load_table"],[34,4,1,"","save_table"]],"mergernet.core.utils.SingletonMeta":[[29,2,1,"","_instances"],[29,2,1,"","_lock"],[29,3,1,"","mro"]],"mergernet.core.utils.Timming":[[30,3,1,"","_format_time"],[30,3,1,"","duration"],[30,3,1,"","end"],[30,3,1,"","start"]],"mergernet.jobs":[[36,0,0,"-","j001_test_train"],[38,0,0,"-","j002_hyperparam"]],"mergernet.jobs.j001_test_train":[[37,1,1,"","Job"]],"mergernet.jobs.j001_test_train.Job":[[37,2,1,"","description"],[37,3,1,"","get_system_resources"],[37,2,1,"","jobid"],[37,2,1,"","name"],[37,3,1,"","post_run"],[37,3,1,"","pre_run"],[37,3,1,"","run"],[37,3,1,"","start_execution"]],"mergernet.jobs.j002_hyperparam":[[39,1,1,"","Job"]],"mergernet.jobs.j002_hyperparam.Job":[[39,2,1,"","description"],[39,3,1,"","get_system_resources"],[39,2,1,"","jobid"],[39,2,1,"","name"],[39,3,1,"","post_run"],[39,3,1,"","pre_run"],[39,3,1,"","run"],[39,3,1,"","start_execution"]],"mergernet.model":[[41,0,0,"-","baseline"],[44,0,0,"-","callback"],[46,0,0,"-","hypermodel"],[49,0,0,"-","plot"],[55,0,0,"-","preprocessing"]],"mergernet.model.baseline":[[42,1,1,"","ConvolutionalClassifier"],[43,1,1,"","Metamodel"]],"mergernet.model.baseline.ConvolutionalClassifier":[[42,3,1,"","compile_model"],[42,3,1,"","train"]],"mergernet.model.callback":[[45,1,1,"","DeltaStopping"]],"mergernet.model.callback.DeltaStopping":[[45,3,1,"","_implements_predict_batch_hooks"],[45,3,1,"","_implements_test_batch_hooks"],[45,3,1,"","_implements_train_batch_hooks"],[45,2,1,"","_keras_api_names"],[45,2,1,"","_keras_api_names_v1"],[45,3,1,"","on_batch_begin"],[45,3,1,"","on_batch_end"],[45,3,1,"","on_epoch_begin"],[45,3,1,"","on_epoch_end"],[45,3,1,"","on_predict_batch_begin"],[45,3,1,"","on_predict_batch_end"],[45,3,1,"","on_predict_begin"],[45,3,1,"","on_predict_end"],[45,3,1,"","on_test_batch_begin"],[45,3,1,"","on_test_batch_end"],[45,3,1,"","on_test_begin"],[45,3,1,"","on_test_end"],[45,3,1,"","on_train_batch_begin"],[45,3,1,"","on_train_batch_end"],[45,3,1,"","on_train_begin"],[45,3,1,"","on_train_end"],[45,3,1,"","set_model"],[45,3,1,"","set_params"]],"mergernet.model.hypermodel":[[47,1,1,"","BayesianTuner"],[48,1,1,"","SimpleHyperModel"]],"mergernet.model.hypermodel.BayesianTuner":[[47,3,1,"","_build_and_fit_model"],[47,3,1,"","_build_hypermodel"],[47,3,1,"","_checkpoint_model"],[47,3,1,"","_configure_tensorboard_dir"],[47,3,1,"","_deepcopy_callbacks"],[47,3,1,"","_delete_checkpoint"],[47,3,1,"","_get_checkpoint_dir"],[47,3,1,"","_get_checkpoint_fname"],[47,3,1,"","_get_tensorboard_dir"],[47,3,1,"","_get_tuner_fname"],[47,3,1,"","_override_compile_args"],[47,3,1,"","_populate_initial_space"],[47,3,1,"","_try_build"],[47,3,1,"","get_best_hyperparameters"],[47,3,1,"","get_best_models"],[47,3,1,"","get_state"],[47,3,1,"","get_trial_dir"],[47,3,1,"","load_model"],[47,3,1,"","on_batch_begin"],[47,3,1,"","on_batch_end"],[47,3,1,"","on_epoch_begin"],[47,3,1,"","on_epoch_end"],[47,3,1,"","on_search_begin"],[47,3,1,"","on_search_end"],[47,3,1,"","on_trial_begin"],[47,3,1,"","on_trial_end"],[47,5,1,"","project_dir"],[47,3,1,"","reload"],[47,5,1,"","remaining_trials"],[47,3,1,"","results_summary"],[47,3,1,"","run_trial"],[47,3,1,"","save"],[47,3,1,"","save_model"],[47,3,1,"","search"],[47,3,1,"","search_space_summary"],[47,3,1,"","set_state"]],"mergernet.model.hypermodel.SimpleHyperModel":[[48,3,1,"","_prepare_data"],[48,3,1,"","build"],[48,3,1,"","fit"]],"mergernet.model.plot":[[50,1,1,"","Serie"],[51,4,1,"","color_color"],[52,4,1,"","data_distribution"],[53,4,1,"","roc"],[54,4,1,"","train_metrics"]],"mergernet.model.plot.Serie":[[50,3,1,"","get_serie"],[50,3,1,"","get_std"],[50,3,1,"","has_std"]],"mergernet.model.preprocessing":[[56,4,1,"","load_jpg"],[57,4,1,"","normalize_rgb"],[58,4,1,"","one_hot"],[59,4,1,"","standardize_rgb"]],"mergernet.services":[[61,0,0,"-","github"],[63,0,0,"-","google"],[65,0,0,"-","imaging"],[67,0,0,"-","legacy"],[69,0,0,"-","sdss"],[71,0,0,"-","splus"],[75,0,0,"-","tensorboard"],[77,0,0,"-","utils"]],"mergernet.services.github":[[62,1,1,"","GithubService"]],"mergernet.services.github.GithubService":[[62,3,1,"","_encode_content"],[62,3,1,"","_get_url"],[62,3,1,"","commit"],[62,3,1,"","get_lastest_job_run"],[62,3,1,"","list_dir"],[62,2,1,"","repo"],[62,2,1,"","token"],[62,2,1,"","user"]],"mergernet.services.google":[[64,1,1,"","GDrive"]],"mergernet.services.google.GDrive":[[64,2,1,"","base_path"],[64,3,1,"","get"],[64,3,1,"","get_url"],[64,3,1,"","is_mounted"],[64,3,1,"","send"],[64,3,1,"","send_dir"]],"mergernet.services.imaging":[[66,1,1,"","BaseImagingService"]],"mergernet.services.imaging.BaseImagingService":[[66,2,1,"","_abc_impl"],[66,3,1,"","batch_download_rgb"],[66,3,1,"","download_rgb"]],"mergernet.services.legacy":[[68,1,1,"","LegacyService"]],"mergernet.services.legacy.LegacyService":[[68,3,1,"","batch_download_rgb"],[68,3,1,"","download_rgb"]],"mergernet.services.sdss":[[70,1,1,"","SloanService"]],"mergernet.services.sdss.SloanService":[[70,3,1,"","batch_download_rgb"],[70,3,1,"","download_rgb"],[70,3,1,"","get_image_filename"]],"mergernet.services.splus":[[72,1,1,"","ImageType"],[73,1,1,"","SplusService"],[74,4,1,"","update_authorization"]],"mergernet.services.splus.ImageType":[[72,2,1,"","fits"],[72,2,1,"","lupton"],[72,2,1,"","trilogy"]],"mergernet.services.splus.SplusService":[[73,3,1,"","_batch_download"],[73,3,1,"","_download_image"],[73,3,1,"","_get_url"],[73,2,1,"","_lock"],[73,3,1,"","_track_tap_job"],[73,3,1,"","batch_image_download"],[73,3,1,"","batch_query"],[73,3,1,"","download_fits"],[73,3,1,"","download_lupton_rgb"],[73,3,1,"","download_trilogy_rgb"],[73,3,1,"","get_image_filename"],[73,3,1,"","query"],[73,3,1,"","update_token"]],"mergernet.services.tensorboard":[[76,1,1,"","TensorboardService"]],"mergernet.services.tensorboard.TensorboardService":[[76,3,1,"","upload_assets"]],"mergernet.services.utils":[[78,4,1,"","append_query_params"],[79,4,1,"","batch_download_file"],[80,4,1,"","download_file"]],mergernet:[[0,0,0,"-","core"],[35,0,0,"-","jobs"],[40,0,0,"-","model"],[60,0,0,"-","services"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","function","Python function"],"5":["py","property","Python property"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:function","5":"py:property"},terms:{"0":[17,42,45,47,48,53,66,68,70,73],"00":84,"0001":[42,47],"02":84,"03":84,"05":[48,84],"09":84,"1":[37,47,48],"10":47,"12":[42,48],"128":[15,42,48,73],"15":[17,73],"2":[17,39,45,47,79],"2022":84,"256":[48,66,68,70],"27":68,"3":[5,16,24,42,48,73],"32":42,"36":84,"4":[15,48],"5":[5,24],"50":16,"55":[66,70],"5e":48,"6":47,"64":48,"7":45,"70":15,"8":73,"abstract":4,"boolean":47,"class":[1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,28,29,30,36,37,38,39,41,42,43,44,45,46,47,48,49,50,53,61,62,63,64,65,66,67,68,69,70,71,72,73,75,76],"default":[5,33,34,47],"do":47,"enum":72,"final":47,"float":[8,15,17,42,47,48,66,68,70,73],"function":[4,16,28,45,47,49,55,67,68,71,73,77],"int":[5,7,8,11,15,42,48,53,62,66,68,70,73,79],"public":73,"return":[29,32,45,47],"static":[5,8,64],"true":[2,8,17,33,34,42,47,53],A:[45,47],For:[45,47],If:[47,53],It:47,NOT:47,The:[4,47,53,68],Will:47,_abc_data:66,_abc_impl:66,_batch_download:73,_build_and_fit_model:47,_build_hypermodel:47,_checkpoint_model:47,_configure_tensorboard_dir:47,_deepcopy_callback:47,_delete_checkpoint:47,_detect_img_extens:5,_discretize_label:5,_download_imag:73,_encode_cont:62,_format_tim:30,_get_checkpoint_dir:47,_get_checkpoint_fnam:47,_get_tensorboard_dir:47,_get_tuner_fnam:47,_get_url:[62,73],_implements_predict_batch_hook:45,_implements_test_batch_hook:45,_implements_train_batch_hook:45,_instanc:29,_keras_api_nam:45,_keras_api_names_v1:45,_lock:[2,13,29,73],_override_compile_arg:47,_populate_initial_spac:47,_prepare_data:48,_thread:[2,13,29],_track_tap_job:73,_try_build:47,_upload_gdr:2,_upload_github:2,abc:66,abov:47,access:[29,47],accuraci:45,action:45,activ:47,adam:42,adapt:[16,47],addit:47,after:47,aggreg:45,alia:45,all:[4,47,53],allocate_lock:[2,13,29],allow_new_entri:47,alpha:47,also:[45,47],an:[45,47,72],ani:[45,47],anyth:47,api:83,ar:[45,47,68],archive_path:6,archive_url:6,arg:[2,13,45,47,68],argument:[45,47],arrai:[31,32],arraydict:17,artifact_path:2,ascens:68,assert:53,astronom:67,attribut:[2,5,37,39,47,62,64,72],author:16,autokera:47,averag:[47,53],b:16,b_band:73,b_fit:15,b_path:15,backward:45,band:[68,73],base:[2,5,6,7,8,10,11,13,15,17,29,30,37,39,42,43,45,47,48,50,62,64,66,68,70,72,73,76],base_path:64,basecontext:73,basejob:[37,39],basepath:70,batch:[45,47],batch_download_rgb:[66,68,70],batch_image_download:73,batch_queri:73,batch_siz:[42,48],bayesian:47,bayesianoptim:47,befor:47,begin:[45,47],best:[47,52,54],best_hp:47,best_step:47,beta:47,bin:[5,7,8],black:47,blind_seri:52,bool:[2,5,6,8,17,33,34,42,48,62,64,70,73,79,80],box:47,branch:62,build:[47,48],call:[45,47],callback:47,can:[4,47,53],cardoso:16,chang:[16,45],check:5,checkpoint:47,class_column:5,code:16,coe:16,color:17,colorsatfac:17,com:16,commit:62,compat:45,compil:45,compile_model:42,complementari:4,comput:53,compute_class_weight:5,compute_max_bin:8,concat_fold_column:5,condit:47,conditional_scop:47,config:[2,5],configu:47,configur:[5,6],contain:47,content:62,context:73,convert:32,coordin:68,core:[37,39,42,48,83],correspond:47,creat:5,credit:16,csv:73,current:[45,47],curv:53,dan:16,data:[2,5,23,27,34,45,47,62,67],data_aug:42,data_path:[5,10,11,37,39],datafram:[5,33,34],dataset:[42,47,48],dataset_op:5,datasetconfig:5,datasetv2:5,datasort:24,date:[81,84],datetim:30,dcoe:16,dec:[15,66,68,70,73],declin:68,deep:4,def:47,defin:[4,67,68],dense_2:48,dense_3:48,dense_block:48,dense_lay:42,dense_units_1:48,dense_units_2:48,dense_units_3:48,densenet201:42,descript:[37,39,76,81,84],desir:[32,68],destin:5,detect_img_extens:6,determin:[45,47],develop:4,df:5,dict:[5,6,8,15,45,62,73,78],dictionari:47,differ:47,directori:47,displai:47,distribut:[7,8],download:[5,67,68],download_arg:73,download_fit:73,download_funct:73,download_legacy_rgb:68,download_lupton_rgb:73,download_rgb:[66,68,70],download_trilogy_rgb:73,dr7objid:70,dr8objid:70,dr9:68,dropout_rate_1:48,dropout_rate_2:48,dropout_rate_3:48,ds_split:5,ds_type:5,dt:30,durat:30,dure:[29,45,47],each:[45,47,53],edu:16,end:[30,45,47],endfor:81,endif:81,engin:[2,48],enumer:72,epoch:[42,45,47,48],error_seri:54,evalu:[45,47],everi:45,exampl:[45,47,53],execut:47,exist:5,experi:83,extend:47,extens:70,f378:73,f395:73,f3d4fc:84,f410:73,f430:73,f515:73,f660:73,f861:73,f:74,fals:[5,6,47,48,50,51,62,70,73,79,80],fetch:11,fig_siz:15,file:[5,32,47,68],fileid:64,filenam:[2,51,52,53,54],find:47,first:29,fit:[32,45,47,48,72],fit_arg:47,fit_kwarg:47,flip:42,fmt:73,fname:2,fname_column:5,fold:[5,6],fold_column:6,folder:5,found:47,frame:[5,33,34],from:[5,32,47,67],from_byt:62,full:47,futur:45,g:73,g_band:73,g_fit:15,g_path:15,gdrive:2,gdrive_path:2,gener:[4,7,47],get:64,get_arrai:17,get_best_hyperparamet:47,get_best_model:47,get_fold:5,get_image_filenam:[70,73],get_lastest_job_run:62,get_logg:13,get_seri:50,get_stat:47,get_std:50,get_system_resourc:[10,37,39],get_trial_dir:47,get_url:64,github:[2,16],given:47,grz:68,gustavo:16,has_std:50,have:68,height:[66,68,70],high:[4,5],histori:47,hp:47,html:16,http:16,http_client:80,hyperparamet:47,i:73,id:[6,47,81,84],imag:[17,32,68],imagenet:[42,48],images_path:6,imagetyp:73,img_typ:73,implement:29,in_memori:5,incept:42,inclu:32,includ:[47,53],index:[45,83],inform:47,initi:47,inplac:17,input:47,input_shap:[42,48],instanc:47,integ:[45,47],intermedi:47,intro:16,is_dataset_download:5,is_mount:64,its:47,job:[81,83],job_artifact:81,job_dat:81,job_descript:81,job_id:11,job_log:81,jobid:[37,39,62],jpg:[32,70],json:84,k:[19,20,26],karg:47,keep:47,kei:[45,47],kera:[2,45,47,48],keras_tun:47,keyword:47,kwarg:[2,10,11,13,37,39,47,50,66,68,70,73],label:[50,53],label_map:6,last:45,layer:[47,68],learn:4,learning_r:[42,48],legend_loc:54,len:53,length:68,less:16,level:[4,5,23],like:67,list:[8,47,53,66,68,70,73,79],list_dir:62,list_job:11,load:[4,32,47],load_model:47,loc:52,local:[32,64],lock:[2,13,29,73],log:[45,47],logdir:[47,76],loss:45,ls:68,lupton:[72,73],m:[16,21],m_max:21,m_min:21,macro:53,mai:[45,47],main:4,make_trilogy_fit:15,mandatori:68,max_rang:52,max_trial:47,maximum:68,mean_seri:54,median:50,memori:16,method:[2,5,7,8,10,11,13,15,17,29,30,37,39,42,45,47,48,50,62,64,66,68,70,73,76],metric:[45,47],micro:53,min_rang:52,miss:5,mode:45,model:[2,4,83],modul:[0,4,35,40,60,67,83],mro:29,multipl:47,multiprocess:73,must:68,n:[24,45,53],n_class:[8,53],n_sigma:24,n_split:[5,8],name:[6,37,39,47,76,81],natanael:16,ndarrai:[5,17,32,70],need:[32,47],nmcardoso:16,nois:73,noiselum:17,none:[2,5,6,10,17,21,31,37,39,42,45,47,48,50,51,52,53,54,62,64,66,68,70,73,80],note:[45,53],npy:32,npz:32,num_initial_point:47,num_model:47,num_trial:47,number:[47,53],numpi:[5,17,32,70],o:16,object:[2,5,6,7,8,10,11,13,15,17,29,30,42,43,47,48,50,62,64,66,68,70,73,76],obtain:47,on_batch_begin:[45,47],on_batch_end:[45,47],on_epoch_begin:[45,47],on_epoch_end:[45,47],on_predict_batch_begin:45,on_predict_batch_end:45,on_predict_begin:45,on_predict_end:45,on_search_begin:47,on_search_end:47,on_test_batch_begin:45,on_test_batch_end:45,on_test_begin:45,on_test_end:45,on_train_batch_begin:45,on_train_batch_end:45,on_train_begin:45,on_train_end:45,on_trial_begin:47,on_trial_end:47,one:47,onli:45,op:5,opt:[66,70],optim:[42,47],option:[2,5,6,10,37,39,42,47,48,62,64,66,68,70,73,80],oracl:47,order:29,origin:16,other:[4,47],output:45,overrid:[45,47],owner:73,page:83,panda:[5,33,34],param:[6,45,73],paramet:[5,32,53,68],pass:[45,47],password:73,path:[2,5,6,10,15,17,32,33,34,37,39,62,64,66,68,70,73,76,79,80],pathdict:17,pathlib:[2,5,6,10,15,32,33,34,37,39,64,66,68,70,73,76,79,80],pathstack:17,perform:[4,45,47],pipelin:47,pixel:68,pixscal:68,plote:53,plu:[16,67],png:32,popul:47,posit:47,posixpath:5,post_run:[10,37,39],pp:27,pre_run:[10,37,39],pred_seri:53,predict:[45,53],prefix:[31,45],preprocess:47,pretrainded_weight:48,pretrained_arch:42,pretrained_weight:42,previous:[4,47],print:47,process:[4,47],program:16,project:47,project_dir:47,properti:47,provid:45,pure:67,python:[5,16,47],q:73,queri:[47,73],query_param:78,r:73,r_band:73,r_column:5,r_fit:15,r_path:15,ra:[15,66,68,70,73],readi:47,recommend:47,reduc:16,refer:83,reinstanti:47,rel:51,reli:47,reload:47,remain:47,remaining_tri:47,remot:64,remov:47,rep:53,repetit:53,replac:[70,73,79,80],repo:62,report:47,represent:[4,5],request:80,resolut:29,resourc:[5,67],restor:47,result:[45,47],results_summari:47,resum:47,retrain:47,retriev:67,rewrot:16,rgb:[5,18,19,68],rgb_config:5,right:68,rmsprop:42,rotat:42,rout:[62,73],rule:4,run:[10,37,39,45,47,81,84],run_job:11,run_trial:47,runid:81,s:[16,29,45,47,67],safe:29,same:68,sampl:42,satperc:17,satur:73,save:[47,68],save_json:2,save_model:47,save_path:[15,66,68,70,73,79,80],savefig:17,scale:[66,68,70],schwarz:16,schwarzam:16,scope:73,sdss:[15,67],search:[47,83],search_space_summari:47,see:47,seed:47,self:47,send:64,send_dir:64,seri:53,serial:47,serializ:47,servic:83,session:80,set:[45,47],set_model:45,set_param:45,set_stat:47,sever:32,shape:[47,53],should:[45,47,68],show:53,shuffl:8,singl:68,singleton:29,size:73,sky:68,sky_siz:15,sort:47,sourc:[2,5,6,7,8,10,11,13,15,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,37,39,42,43,45,47,48,50,51,52,53,54,56,57,58,59,62,64,66,68,70,72,73,74,76,78,79,80],source_dict:15,space:47,spawn:68,specif:53,specifi:5,split:[5,7],split_al:8,split_id:8,sql:73,stamp:68,stampsrgb:22,start:[30,45],start_execut:[10,37,39],state:47,std:53,step:47,steps_per_execut:45,stop:47,storag:32,store:68,str:[2,5,6,10,11,15,30,32,33,34,37,39,42,48,62,64,66,68,70,73,76,78,79,80],stretch:73,stsci:16,subclass:45,subroutin:47,summari:47,support:32,survei:[67,68],synchron:29,table_:51,table_path:6,table_url:6,task:4,tensorflow:5,test:[37,39,45],test_seri:52,text:73,tf:[45,47],tfrecord:4,thi:[4,45,47,67],thread:[29,68],time:47,timedelta:30,token:62,track:47,train:[2,42,45,47,48],train_seri:52,trial:47,trial_id:47,trilogi:72,trilogy_fits_to_png:15,true_seri:53,tune:47,tune_new_entri:47,tuner:47,tupl:[5,42,48,70],type:[5,15,29,32,47],u:73,union:[2,5,10,32,33,34,37,39,64,70,73,76,79,80],unlock:[2,13,29],unsatperc:22,until:45,untrain:47,up:45,update_token:73,upload:2,upload_asset:76,upload_dir:2,upload_json:2,upload_log:2,upload_model:2,upload_text:2,url:[78,79,80],us:[4,29,47],usag:16,use_gdr:2,use_github:2,user:62,usernam:73,utilitari:67,val_:45,valid:[45,47],valu:[45,47,53,72],verbos:42,via:47,vline:52,wai:16,web:[5,67],weight:47,well:4,when:47,where:[53,68],whether:47,which:[4,47],who:4,width:[66,68,70],within:[45,47],worker:[66,68,70,73,79],worst:47,www:16,x:[8,25,47,56,57,58,59],x_column:6,x_column_suffix:6,xlabel:52,xlim:[51,54],y1:23,y:[5,8,47,56,57,58,59],y_column:6,ylabel:52,ylim:[51,54],you:47,your:47,z:73,zoom_po:53,zoom_rang:53},titles:["mergernet.core","mergernet.core.artifacts","mergernet.core.artifacts.ArtifactHelper","mergernet.core.constants","mergernet.core.dataset","mergernet.core.dataset.Dataset","mergernet.core.dataset.DatasetConfig","mergernet.core.dataset.DistributionKFold","mergernet.core.dataset.StratifiedDistributionKFold","mergernet.core.jobs","mergernet.core.jobs.BaseJob","mergernet.core.jobs.JobRunner","mergernet.core.logger","mergernet.core.logger.Logger","mergernet.core.rgb","mergernet.core.rgb.RGB","mergernet.core.trilogy","mergernet.core.trilogy.MakeImg","mergernet.core.trilogy.RGB2im","mergernet.core.trilogy.adjust_saturation","mergernet.core.trilogy.da","mergernet.core.trilogy.get_clip","mergernet.core.trilogy.get_levels","mergernet.core.trilogy.imscale","mergernet.core.trilogy.meanstd","mergernet.core.trilogy.rms","mergernet.core.trilogy.satK2m","mergernet.core.trilogy.setLevel","mergernet.core.utils","mergernet.core.utils.SingletonMeta","mergernet.core.utils.Timming","mergernet.core.utils.array_fallback","mergernet.core.utils.load_image","mergernet.core.utils.load_table","mergernet.core.utils.save_table","mergernet.jobs","mergernet.jobs.j001_test_train","mergernet.jobs.j001_test_train.Job","mergernet.jobs.j002_hyperparam","mergernet.jobs.j002_hyperparam.Job","mergernet.model","mergernet.model.baseline","mergernet.model.baseline.ConvolutionalClassifier","mergernet.model.baseline.Metamodel","mergernet.model.callback","mergernet.model.callback.DeltaStopping","mergernet.model.hypermodel","mergernet.model.hypermodel.BayesianTuner","mergernet.model.hypermodel.SimpleHyperModel","mergernet.model.plot","mergernet.model.plot.Serie","mergernet.model.plot.color_color","mergernet.model.plot.data_distribution","mergernet.model.plot.roc","mergernet.model.plot.train_metrics","mergernet.model.preprocessing","mergernet.model.preprocessing.load_jpg","mergernet.model.preprocessing.normalize_rgb","mergernet.model.preprocessing.one_hot","mergernet.model.preprocessing.standardize_rgb","mergernet.services","mergernet.services.github","mergernet.services.github.GithubService","mergernet.services.google","mergernet.services.google.GDrive","mergernet.services.imaging","mergernet.services.imaging.BaseImagingService","mergernet.services.legacy","mergernet.services.legacy.LegacyService","mergernet.services.sdss","mergernet.services.sdss.SloanService","mergernet.services.splus","mergernet.services.splus.ImageType","mergernet.services.splus.SplusService","mergernet.services.splus.update_authorization","mergernet.services.tensorboard","mergernet.services.tensorboard.TensorboardService","mergernet.services.utils","mergernet.services.utils.append_query_params","mergernet.services.utils.batch_download_file","mergernet.services.utils.download_file","#{{ jobid }}: {{ job_name }}","Experiments","Welcome to mergernet\u2019s documentation!","#2: Test job","API Reference"],titleterms:{"2":84,adjust_satur:19,api:85,append_query_param:78,array_fallback:31,artifact:[1,2,81,84],artifacthelp:2,baseimagingservic:66,basejob:10,baselin:[41,42,43],batch_download_fil:79,bayesiantun:47,callback:[44,45],color_color:51,constant:3,content:83,convolutionalclassifi:42,core:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34],da:20,data_distribut:52,dataset:[4,5,6,7,8],datasetconfig:6,deltastop:45,distributionkfold:7,document:83,download_fil:80,experi:82,gdrive:64,get_clip:21,get_level:22,github:[61,62],githubservic:62,googl:[63,64],hypermodel:[46,47,48],imag:[65,66],imagetyp:72,imscal:23,indic:83,j001_test_train:[36,37],j002_hyperparam:[38,39],job:[9,10,11,35,36,37,38,39,84],job_nam:81,jobid:81,jobrunn:11,legaci:[67,68],legacyservic:68,load_imag:32,load_jpg:56,load_tabl:33,log:81,logger:[12,13],makeimg:17,meanstd:24,mergernet:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,83],metamodel:43,model:[40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59],normalize_rgb:57,one_hot:58,plot:[49,50,51,52,53,54],preprocess:[55,56,57,58,59],refer:85,rgb2im:18,rgb:[14,15],rm:25,roc:53,s:83,satk2m:26,save_t:34,sdss:[69,70],seri:50,servic:[60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80],setlevel:27,simplehypermodel:48,singletonmeta:29,sloanservic:70,splu:[71,72,73,74],splusservic:73,standardize_rgb:59,stratifieddistributionkfold:8,tabl:83,tensorboard:[75,76],tensorboardservic:76,test:84,tim:30,train_metr:54,trilogi:[16,17,18,19,20,21,22,23,24,25,26,27],update_author:74,util:[28,29,30,31,32,33,34,77,78,79,80],welcom:83}})