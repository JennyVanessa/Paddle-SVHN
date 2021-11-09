import lmdb
import example_pb2

path_to_lmdb_dir = './data/train.lmdb'
reader = lmdb.open(path_to_lmdb_dir)
txn = reader.begin()
cursor = txn.cursor()

cursor.next()
key, value = cursor.item()
example = example_pb2.Example()
example.ParseFromString(value)
print('key:', key.decode())

image = np.frombuffer(example.image, dtype=np.uint8)
length = example.length
digits = example.digits

print('length: %d, digits: %d,%d,%d,%d,%d' % (
    length, digits[0], digits[1], digits[2], digits[3], digits[4]))

imshow(image.reshape([64, 64, 3]))
