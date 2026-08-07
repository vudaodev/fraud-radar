"""
Shared feature transforms. Used by training notebooks and the pickled pipeline.
"""


def secs_to_hour_of_day(sec_from_start):
    """Input: seconds since the start of the dataset
    Output: hour of the day (0-24)
    """
    return (sec_from_start % 86400) / 3600
