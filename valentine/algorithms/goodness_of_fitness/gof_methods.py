import numpy as np
from scipy import stats

def remove_outliers(data):
    """
    A simple function to remove outliers using the Interquartile Range (IQR) method.
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]


# === GOF Methods

def ks_test(data1, data2, hist_bins=10):
    """
    Performs a two-sample Kolmogorov-Smirnov (KS) test to check if two
    independent samples are drawn from the same continuous distribution.

    Args:
        data1 (list or np.array): The observed data for the hypothesized distribution.
        data2 (list or np.array): The observed data to be tested against the hypothesized distribution.

    Returns:
        scipy.stats.ks_2samp: An object containing the KS statistic and the p-value.

    Null hypothesis:
        The two samples are drawn from the same continuous distribution.

    Observations:
        The KS test is a non-parametric test, meaning it doesn't assume the data
        follows a specific distribution (like a normal distribution). It is
        sensitive to differences in both location (median) and shape of the
        empirical cumulative distribution functions (ECDFs) of the two samples.
    """

    # --- Previous data treatment ---

    # Robustly handle None values and zeros before converting to numpy array
    # A list comprehension is ideal for filtering out None and NaN values
    data1 = [val for val in data1 if val is not None and val == val]
    data2 = [val for val in data2 if val is not None and val == val]

    # Convert to numpy arrays for efficient operations
    data1 = np.array(data1, dtype=int)
    data2 = np.array(data2, dtype=int)

    # Remove outliers
    data1 = remove_outliers(data1)
    data2 = remove_outliers(data2)

    # Create histograms and use bin counts
    counts1, _ = np.histogram(data1, bins=hist_bins)
    counts2, _ = np.histogram(data2, bins=hist_bins)

    dist1 = counts1 / np.sum(counts1) * 100
    dist2 = counts2 / np.sum(counts2) * 100
    
    # --- Execute the test ---
    return stats.ks_2samp(dist1, dist2)


def ad_test(data1, data2, hist_bins=10):
    """
    Performs the Anderson-Darling test for goodness-of-fit to check if a
    sample is drawn from a specific continuous distribution.

    Args:
        data1 (list or np.array): The observed data for the hypothesized distribution.
        data2 (list or np.array): The observed data to be tested against the hypothesized distribution.

    Returns:
        scipy.stats.anderson: An object containing the AD statistic, critical values, and significance levels.

    Null hypothesis:
        The sample is drawn from the specified distribution.

    Observations:
        The Anderson-Darling test is a modification of the KS test.
        It gives more weight to the tails of the distribution, making it more
        sensitive than the KS test for detecting deviations in the tails.
    """

    # --- Previous data treatment ---

    # Robustly handle None values and zeros before converting to numpy array
    # A list comprehension is ideal for filtering out None and NaN values    
    data1 = [val for val in data1 if val is not None and val == val]
    data2 = [val for val in data2 if val is not None and val == val]

    # Convert to numpy arrays for efficient operations
    data1 = np.array(data1, dtype=int)
    data2 = np.array(data2, dtype=int)

    # Remove outliers
    data1 = remove_outliers(data1)
    data2 = remove_outliers(data2)

    # Create histograms and use bin counts
    counts1, _ = np.histogram(data1, bins=hist_bins)
    counts2, _ = np.histogram(data2, bins=hist_bins)

    dist1 = counts1 / np.sum(counts1) * 100
    dist2 = counts2 / np.sum(counts2) * 100
    
    # --- Execute the test ---
    return stats.anderson_ksamp([dist1, dist2], method=stats.PermutationMethod(n_resamples=9999, random_state=42))


def chisq_test(data1, data2):
    """
    Performs a Chi-Squared goodness-of-fit test to check if a variable
    follows a hypothesized distribution.
    
    Args:
        data1 (list or np.array): The observed data for the hypothesized distribution.
        data2 (list or np.array): The observed data to be tested against the hypothesized distribution.
    
    Returns:
        scipy.stats.chisquare: An object containing the chi-squared statistic and the p-value.

    Null hypothesis: 
        The count numbers are sampled from a multinomial distribution

    Observations:
        There exists three types of Chi-Squared tests, the goodness-of-fit, homogeneity and independence test
        This function performs the GoF test
    """
    
    # --- Previous data treatment

    # Convert data to NumPy arrays for easier manipulation
    data1 = np.array(data1, dtype='float64')
    data2 = np.array(data2, dtype='float64')

    # Convert boolean data to numeric (0 or 1)
    if data1.dtype == 'bool':
        data1 = data1.astype(int)
    if data2.dtype == 'bool':
        data2 = data2.astype(int)
    
    # Replace NULL values (NaN) with a new category
    max_value = np.nanmax(np.concatenate([data1, data2])) + 1
    data1 = np.nan_to_num(data1, nan=max_value)
    data2 = np.nan_to_num(data2, nan=max_value)
    
    # Find common categories and create a union of all unique values
    common_values = np.intersect1d(np.unique(data1), np.unique(data2))
    
    # Create frequency tables and normalize them
    expected_counts = np.array([np.sum(data1 == val) for val in common_values])
    expected_freq = expected_counts / np.sum(expected_counts)

    new_counts = np.array([np.sum(data2 == val) for val in common_values])
    new_freq = new_counts / np.sum(new_counts)

    # --- Execute the test
    
    # Run the chi-squared test. The function returns the chi-squared statistic and the p-value.
    return stats.chisquare(f_obs=new_freq, f_exp=expected_freq)


def g_test(data1, data2):
    """
    Performs a G-test on two datasets.
    
    Args:
        data1 (list or np.array): The observed data for the hypothesized distribution.
        data2 (list or np.array): The observed data to be tested against the hypothesized distribution.

    Returns:
        float: The G-statistic value.
        float: The p-value of the test.

    Null hypothesis:
        The observed frequencies result from random sampling from a 
        distribution with the given expected frequencies.

    Observations:
        Scipy does not have a explicit g-test function,
        but it has an example of g-test in the source code
        >>> power_divergence([16, 18, 16, 14, 12, 12],
        ...                  f_exp=[16, 16, 16, 16, 16, 8],
        ...                  lambda_='log-likelihood')
        (3.3281031458963746, 0.6495419288047497)
    """
    
    # --- Previous data treatment
    
    # Convert data to NumPy arrays for easier manipulation
    data1 = np.array(data1, dtype='float64')
    data2 = np.array(data2, dtype='float64')

    # Convert boolean data to numeric (0 or 1)
    if data1.dtype == 'bool':
        data1 = data1.astype(int)
    if data2.dtype == 'bool':
        data2 = data2.astype(int)
 
    # Replace NULL values (NaN) with a new category
    max_value = np.nanmax(np.concatenate([data1, data2])) + 1
    data1 = np.nan_to_num(data1, nan=max_value)
    data2 = np.nan_to_num(data2, nan=max_value)
    
    # Find common categories
    common_values = np.intersect1d(np.unique(data1), np.unique(data2))
    
    # Create frequency tables and normalize them
    expected_counts = np.array([np.sum(data1 == val) for val in common_values])
    expected_freq = expected_counts / np.sum(expected_counts)

    new_counts = np.array([np.sum(data2 == val) for val in common_values])
    new_freq = new_counts / np.sum(new_counts)

    # --- Execute the test

    return stats.power_divergence(f_obs=new_freq, f_exp=expected_freq, lambda_='log-likelihood')

def encode_categories(data1, data2):
    combined = np.concatenate([data1, data2])

    # Find unique categories and encode them as integers
    categories, encoded = np.unique(combined, return_inverse=True)

    # Split the encoded data back into the original datasets
    data1_encoded = encoded[:len(data1)]
    data2_encoded = encoded[len(data1):]

    return data1_encoded, data2_encoded, categories