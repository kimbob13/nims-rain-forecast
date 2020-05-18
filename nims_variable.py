__all__ = ['parse_variables', 'read_variable_value', 'get_variable_name']

def parse_variables(variables_args):
    """
    Return list of variables index

    <Parameters>
    variables_args
        - [int]: How many variables to use.
        - [str]: Name of one variable
        - [list[str]]: List of variable names to use

    <Return>
    variables [list[int]]: List of variable index
    """
    variables_dict = {'rain':  0, 'cape':  1, 'cin':  2, 'swe':  3, 'hel': 4,
                      'ct'  :  5, 'vt'  :  6, 'tt' :  7, 'si' :  8, 'ki' : 9,
                      'li'  : 10, 'ti'  : 11, 'ssi': 12, 'pw' : 13}

    if (len(variables_args) == 1) and (variables_args[0].isdigit()):
        # Only one variable is specified by index
        variables = list(range(int(variables_args[0])))

    elif (len(variables_args) == 1) and (variables_args[0] in variables_dict):
        # Only one variable is specified by name
         variables = [variables_dict[variables_args[0]]]

    elif len(variables_args) > 1:
        # List of variable names to use
        variables = set()
        if 'rain' not in variables_args:
            print("You don't add rain variable. It is added by default")
            variables.add(variables_dict['rain'])

        for var_name in variables_args:
            variables.add(variables_dict[var_name])

        variables = sorted(list(variables))

    else:
        # No variable is specified in arguments. Use rain variables by default
        print("You don't specify any variables. Use rain variable by default")
        varaiables = [variables_dict['rain']]

    return variables

def read_variable_value(one_hour_dataset, var_idx):
    """
    Read proper variable based on var_idx.
    For example, if var_idx == 0, it should read 'rain' data,
    and if var_idx == 4, it should read 'hel' data.

    Variable List:
    [0] : rain [1] : cape [2] : cin  [3] : swe [4]: hel
    [5] : ct   [6] : vt   [7] : tt   [8] : si  [9]: ki
    [10]: li   [11]: ti   [12]: ssi  [13]: pw

    <Parameters>
    one_hour_dataset [xarray dataset]: dataset for one hour to extract data
    var_idx [int]: index for variables list

    <Return>
    one_var_data [np.ndarray]: numpy array of value (CHW format)
    """
    assert var_idx >= 0 and var_idx <= 13

    if var_idx == 0:
        one_var_data = one_hour_dataset.rain.values
    elif var_idx == 1:
        one_var_data = one_hour_dataset.cape.values
    elif var_idx == 2:
        one_var_data = one_hour_dataset.cin.values
    elif var_idx == 3:
        one_var_data = one_hour_dataset.swe.values
    elif var_idx == 4:
        one_var_data = one_hour_dataset.hel.values
    elif var_idx == 5:
        one_var_data = one_hour_dataset.ct.values
    elif var_idx == 6:
        one_var_data = one_hour_dataset.vt.values
    elif var_idx == 7:
        one_var_data = one_hour_dataset.tt.values
    elif var_idx == 8:
        one_var_data = one_hour_dataset.si.values
    elif var_idx == 9:
        one_var_data = one_hour_dataset.ki.values
    elif var_idx == 10:
        one_var_data = one_hour_dataset.li.values
    elif var_idx == 11:
        one_var_data = one_hour_dataset.ti.values
    elif var_idx == 12:
        one_var_data = one_hour_dataset.ssi.values
    elif var_idx == 13:
        one_var_data = one_hour_dataset.pw.values

    return one_var_data

def get_variable_name(var_idx):
    """
    Variable List:
    [0] : rain [1] : cape [2] : cin  [3] : swe [4]: hel
    [5] : ct   [6] : vt   [7] : tt   [8] : si  [9]: ki
    [10]: li   [11]: ti   [12]: ssi  [13]: pw
    """
    assert var_idx >= 0 and var_idx <= 13

    if var_idx == 0:
        return 'rain'
    elif var_idx == 1:
        return 'cape'
    elif var_idx == 2:
        return 'cin'
    elif var_idx == 3:
        return 'swe'
    elif var_idx == 4:
        return 'hel'
    elif var_idx == 5:
        return 'ct'
    elif var_idx == 6:
        return 'vt'
    elif var_idx == 7:
        return 'tt'
    elif var_idx == 8:
        return 'si'
    elif var_idx == 9:
        return 'ki'
    elif var_idx == 10:
        return 'li'
    elif var_idx == 11:
        return 'ti'
    elif var_idx == 12:
        return 'ssi'
    elif var_idx == 13:
        return 'pw'