from azureml.core.compute import ComputeTarget, AmlCompute

cluster_name = "udaciy_yum"

# TODO: Create compute cluster
# Use vm_size = "Standard_D2_V2" in your provisioning configuration.
# max_nodes should be no greater than 4.

### YOUR CODE HERE ###
cluster_basic = AmlCompute(
    name="udaciy_ml_yum",
    type="amlcompute",
    size="Standard_D2_V2",
    location="westus",
    min_instances=0,
    max_instances=4
)
ml_client.begin_create_or_update(cluster_basic).result()

--------------------------------
from azureml.widgets import RunDetails
from azureml.train.sklearn import SKLearn
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.parameter_expressions import choice, uniform
from azureml.core import Environment, ScriptRunConfig
from azureml.core.experiment import Experiment
import os

# Specify parameter sampler
# ps = ### YOUR CODE HERE ###
ps = RandomParameterSampling( {
        "learning_rate": uniform(0.05, 0.1),
        "batch_size": choice(16, 32, 64, 128)
    }
)

# Specify a Policy
# policy = ### YOUR CODE HERE ###
policy = BanditPolicy(evaluation_interval = 5, slack_factor=0.1)

if "training" not in os.listdir():
    os.mkdir("./training")

# Setup environment for your training run
sklearn_env = Environment.from_conda_specification(name='sklearn-env', file_path='conda_dependencies.yml')

# Create a ScriptRunConfig Object to specify the configuration details of your training job
# src = ### YOUR CODE HERE ###
# create or load an experiment
experiment = Experiment(workspace, 'LRExperiment')
# create or retrieve a compute target
cluster = workspace.compute_targets['YumCluster']
# create or retrieve an environment
env = Environment.get(ws, name='YumEnvironment')
# configure and submit your training run
src = ScriptRunConfig(source_directory='.',
                        script='train.py',
                        arguments=['--arg1', arg1_val, '--arg2', arg2_val],
                        compute_target=cluster,
                        environment=env)

# Create a HyperDriveConfig using the src object, hyperparameter sampler, and policy.
# hyperdrive_config = ### YOUR CODE HERE ###
hyperdrive_config = HyperDriveConfig(run_config=src,
                                hyperparameter_sampling=ps,
                                policy=policy,
                                primary_metric_name='Accuracy',
                                primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                max_total_runs=20,
                                max_concurrent_runs=4)

-------------------------------------
# Submit your hyperdrive run to the experiment and show run details with the widget.

### YOUR CODE HERE ###
run1 = experiment.submit(config=hyperdrive_config)
RunDetails(run1).show()
run1

-------------------------------------
import joblib
# Get your best run and save the model from that run.

### YOUR CODE HERE ###
# tag the runs 
run1.add_properties("author":"yum")
run1.tag("quality","best_run")
list(experiment.get_runs(properties = {"author":"yum"}, tags = {"quality" : "best_run"}))
best_model = experiment.get_runs(properties = {"author":"yum"}, tags = {"quality" : "best_run"})
-----------------------
from azureml.data.dataset_factory import TabularDatasetFactory

# Create TabularDataset using TabularDatasetFactory
# Data is available at: 
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

### YOUR CODE HERE ###
datastore_path = ['https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv']
dataset = Dataset.Tabular.from_delimited_files(path=datastore_path)
ds = dataset.to_pandas_dataframe() 
    
-------------------------
from train import clean_data

# Use the clean_data function to clean your data.
x, y = clean_data(ds)

----------------------------
from azureml.train.automl import AutoMLConfig

# Set parameters for AutoMLConfig
# NOTE: DO NOT CHANGE THE experiment_timeout_minutes PARAMETER OR YOUR INSTANCE WILL TIME OUT.
# If you wish to run the experiment longer, you will need to run this notebook in your own
# Azure tenant, which will incur personal costs.
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task= 'classification',
    primary_metric= 'Accuracy',
    training_data=x,
    label_column_name=y,
    n_cross_validations=5)
	
--------------------------------
# Submit your automl run

### YOUR CODE HERE ###
autorun = experiment.submit(automl_config, show_output = True)
--------------------------------
# Retrieve and save your best automl model.

### YOUR CODE HERE ###
autorun.add_properties("author":"yum")
autorun.tag("quality","best_auto_run")
list(experiment.get_runs(properties = {"author":"yum"}, tags = {"quality" : "best_auto_run"}))
best_auto_model = experiment.get_runs(properties = {"author":"yum"}, tags = {"quality" : "best_auto_run"})



