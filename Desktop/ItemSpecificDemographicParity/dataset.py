import numpy as np
import random

class Dataset:
    def __init__(self, train_file, test_file, negative_file, user_file, movie_file):
        self.train_file = train_file
        self.test_file = test_file
        self.negative_file = negative_file
        self.user_file = user_file
        self.movie_file = movie_file
        self.user_pool = set()
        self.item_pool = set()

        # Load data
        self.user_attributes = self.load_user_attributes(user_file)
        self.movie_categories = self.load_movie_categories(movie_file)
        self.train_data = self.load_train()
        self.test_ratings, self.test_negatives = self.load_test()

        # Number of users/items
        self.num_users = max(self.user_pool) + 1
        self.num_items = max(self.item_pool) + 1

        # ---- FIX ----
        item_sensitive_vectors = []  # initialize the list

        # iterate over ALL item IDs to match model embedding indexing
        for item in range(self.num_items):
            vec = self.get_item_sensitive_vector(item)
            item_sensitive_vectors.append(vec)

        self.item_sensitive_vectors = np.array(item_sensitive_vectors)

    def load_user_attributes(self, user_file):
        """Load user attributes (age, gender) from users.dat (tab-separated)."""
        user_attributes = {}  # user_id -> (age, gender)

        with open(user_file, 'r', encoding='utf-8-sig') as f:  # handle BOM if present
            for line in f:
                parts = line.strip().split("\t")   # your file uses tabs
                if len(parts) < 3:
                    continue  # skip malformed lines

                user = int(parts[0]) - 1  # convert to 0-based indexing

                # Gender: M=0, F=1
                gender_str = parts[1].strip()
                gender = 0 if gender_str == 'M' else 1

                # Age: MovieLens encodes age as a code (not real years)
                # 1 = under 18 → map to 0 (child)
                # else         → map to 1 (adult)
                age_code = int(parts[2])
                age = 0 if age_code == 1 else 1

                user_attributes[user] = (age, gender)

        print(f"Loaded {len(user_attributes)} users with attributes")
        return user_attributes

    def load_movie_categories(self, movie_file):
        """Load movie categories."""
        movie_categories = {}  # item_id -> category (0 or 1)
        with open(movie_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                parts = line.strip().split("\t")
                item = int(parts[0])  # keep original ID
                try:
                    category = int(parts[1])
                    if category not in [0, 1]:
                        print(f"Invalid category {category} for item {item}")
                        category = 0
                except ValueError:
                    print(f"Invalid category format: {line.strip()}")
                    category = 0
                movie_categories[item] = category

        print(f"Loaded {len(movie_categories)} movies with attributes")
        return movie_categories        

    def load_train(self):
        """Load training data."""
        data = []
        with open(self.train_file, 'r', encoding='utf-8') as f:
            for line in f:
                user, item, *_ = map(int, line.strip().split('\t')[:2])
                data.append((user, item, 1))  # positive label
                self.user_pool.add(user)
                self.item_pool.add(item)
        return data

    def load_test(self):
        """Load test ratings and negatives."""
        test_ratings = []
        test_negatives = []
        with open(self.test_file, 'r', encoding='utf-8') as f:
            for line in f:
                user, item, *_ = map(int, line.strip().split('\t')[:2])
                test_ratings.append((user, item))
                self.user_pool.add(user)
                self.item_pool.add(item)

        with open(self.negative_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = list(map(int, line.strip().split('\t')[1:]))
                negatives = [item for item in parts]  # Adjust to 0-based
                test_negatives.append(negatives)
                for item in negatives:
                    self.item_pool.add(item)

        return test_ratings, test_negatives

    def get_train_instances(self, num_negatives=4):
        """Generate training instances with negative sampling."""
        users, items, labels = [], [], []
        ages, genders, categories = [], [], []

        train_set = set((u, i) for u, i, _ in self.train_data)

        for (user, item, _) in self.train_data:
            # Positive instance
            users.append(user)
            items.append(item)
            labels.append(1)

            age, gender = self.user_attributes.get(user, (0, 0))  # Default if missing
            ages.append(age)
            genders.append(gender)
            category = self.movie_categories.get(item, 0)
            categories.append(category)

            # Negative instances
            for _ in range(num_negatives):
                neg_item = random.randint(0, self.num_items - 1)
                while (user, neg_item) in train_set:
                    neg_item = random.randint(0, self.num_items - 1)

                users.append(user)
                items.append(neg_item)
                labels.append(0)

                age, gender = self.user_attributes.get(user, (0, 0))
                ages.append(age)
                genders.append(gender)
                category = self.movie_categories.get(neg_item, 0)
                categories.append(category)

        return (np.array(users), np.array(items), np.array(labels),
                np.array(ages), np.array(genders), np.array(categories))

    def get_item_sensitive_vector(self,item):
        cat = self.movie_categories.get(item, 0)

        # [M, F, YM, YF, OM, OF]
        if cat == 0:  # gender-only
            return [1, 1, 0, 0, 0, 0]
        else:         # age + gender
            return [0, 0, 1, 1, 1, 1]

    def get_test_instances(self):
        """Generate test instances with user attributes and movie categories."""
        users, items, labels = [], [], []
        ages, genders, categories = [], [], []

        for (user, item) in self.test_ratings:
            users.append(user)
            items.append(item)
            labels.append(1)  # Test ratings are positive
            age, gender = self.user_attributes.get(user, (0, 0))  # Default if missing
            ages.append(age)
            genders.append(gender)
            category = self.movie_categories.get(item, 0)  # Default to 0 (Rated R)
            categories.append(category)

        return (np.array(users), np.array(items), np.array(labels),
                np.array(ages), np.array(genders), np.array(categories))