def filter_channel_options(channel_options, theta_range):
    """
    Filters the channel options based on the given theta range.

    :param channel_options: List of all channel options (e.g., ['channel_option_0', ..., 'channel_option_1'])
    :param theta_range: Tuple specifying the theta range (min_theta, max_theta)
    :return: Filtered list of channel options
    """
    min_theta, max_theta = theta_range
    return [
        option for option in channel_options
        if min_theta <= float(option.split('_')[-1]) <= max_theta
    ]