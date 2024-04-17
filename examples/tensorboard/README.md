# Using Tensorboard with Pytorch

It's important to have the correct version of TensorBoard installed. Otherwise,
we cannot display results saved in `pt.trace.json`, produced by
`on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_log_dir)`.

The following package is included in the `.devcontainer/requirements.txt` file:

```
pip install torch-tb-profiler
```

The Docker container will bind `localhost:6006` to TensorBoard.
After profile is collected you should see a `PYTORCH_PROFILER` tab at the topbar.

Check [tensorboard_profiler_tutorial.ipynb](tensorboard_profiler_tutorial.ipynb)
to generate the profiles.


## Starting TensorBoard WebService


### Option 1

My recommended option is to open the terminal inside the devcontainer and
execute the following command at the root of this project:

```
tensorboard --logdir `pwd`/examples/tensorboard/prof/
```

This method has the advantage of leaving the process in the terminal, which is
easy to kill/terminate.


### Option 2

Open the TensorBoard extension in VSCode: 

`Ctrl+Shift+P` > Python: Launch TensorBoard > Select the workspace root folder
or the folder where logs are stored.

Once TensorBoard starts, it cannot be stopped. This may cause the container to
remain active, so make sure to stop the container after use.