def check_command_line_arguments(in_arg):
    """
    Returns:
    Nothing - just prints to console
    """
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line arguments
        print("Command Line Arguments:\n     data_directory =", in_arg.data_dir,
              "\n    arch =", in_arg.arch, "\n num_epochs =", in_arg.epochs,
              "\n Learning Rate =", in_arg.lr,"\n hidden units =", in_arg.hu)
