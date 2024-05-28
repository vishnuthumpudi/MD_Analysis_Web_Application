import unittest
import numpy as np
import tempfile
from unittest.mock import patch, mock_open
from your_module import main  # replace 'your_module' with the actual module name

class TestMain(unittest.TestCase):
    @patch('builtins.open', new_callable=mock_open, read_data="1,2,3,4,5,6,7,8,9,10,11,12\n"*10)
    def test_main(self, mock_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
            tmp_csv.write(mock_file.read().encode())
            csv_path = tmp_csv.name

        data = np.genfromtxt(csv_path, delimiter=',',skip_header=1,usecols=range(1, 13))
        row_sums = np.sum(data,axis = 1)
        count = 0
        frames_list = []

        for i,row_sum in enumerate(row_sums):
            if(row_sum >= 5):
                count+=1
                frames_list.append(i)

        self.assertEqual(count, 10)
        self.assertEqual(frames_list, list(range(10)))

if __name__ == '__main__':
    unittest.main()