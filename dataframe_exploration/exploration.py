from common.services import print_service


def explore_dataframe(dataframe):
    print_service("Head", dataframe.head(10))
    print_service("Shape", dataframe.shape)
    print_service("Null Values", dataframe.isnull().sum())


