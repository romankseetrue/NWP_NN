from prepare_data import DataLoader

if __name__ == '__main__':
    loader: DataLoader = DataLoader(33345)
    print(loader.get_data())
