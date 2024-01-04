import random

class Hcms:
    """
    Higher-order CMS (H-CMS) data structure implementation for anomaly detection.
    
    Methods
    -------
    hash(self, elem, i) :
        Hash function used to determine the bucket for an element in a specific row.
        
    insert(self, a, b, weight) : 
        Inserts an edge (a, b) with a specified weight into the count matrix.
        
    decay(self, decay_factor) : 
        Applies decay to the entire count matrix to decrease counts over time.
        
    getAnoEdgeGlobalScore(self, src, dst):
        Computes the anomaly score for an edge (src, dst) based on local density.
        
    getAnographScore(self):
        Computes the global anomaly score for the entire count matrix based on local densities.
    """
    def __init__(self, r, b, d=None):
        """
        Parameters
        ----------
        r : int
            Number of rows.
        b : int
            Number of buckets in each row.
        d : int, optional
            Number of dense submatrices. The default is None.

        Returns
        -------
        None.

        """
        
        self.num_rows = r
        self.num_buckets = b
        self.num_dense_submatrices = d
        
        # Initialize hash functions
        self.hash_a = [random.randint(1, self.num_buckets - 1) for _ in range(self.num_rows)]
        self.hash_b = [random.randint(0, self.num_buckets - 1) for _ in range(self.num_rows)]
        
        # Initialize count matrix
        self.count = [[[0 for _ in range(self.num_buckets)] for _ in range(self.num_buckets)] for _ in range(self.num_rows)]

    def hash(self, elem, i):
        """
        Hash function used to determine the bucket for an element in a specific row.

        Parameters
        ----------
        elem : int
            Element to be hashed.
        i : int
            Index of the hash function.

        Returns
        -------
        int
            Hashed bucket index.

        """
        
        # Calculate the hash value using hash coefficients
        resid = (elem * self.hash_a[i] + self.hash_b[i]) % self.num_buckets
        
        # Adjust the hash value if it's negative
        return resid + self.num_buckets if resid < 0 else resid

    def insert(self, a, b, weight):
        """
        Inserts an edge (a, b) with a specified weight into the count matrix.

        Parameters
        ----------
        a : int
            Node representing one end of the edge.
        b : int
            Node representing the other end of the edge.
        weight : float
            Weight associated with the edge.

        Returns
        -------
        None.

        """
        
        # Iterate over the rows to insert the edge into the count matrix
        for i in range(self.num_rows):
            
            # Determine the buckets using the hash function
            a_bucket = self.hash(a, i)
            b_bucket = self.hash(b, i)
            
            # Increment the count for the edge in the count matrix
            self.count[i][a_bucket][b_bucket] += weight

    def decay(self, decay_factor):
        """
        Applies decay to the entire count matrix to decrease counts over time.

        Parameters
        ----------
        decay_factor : float
            Factor by which counts are decayed.

        Returns
        -------
        None.

        """
        
        # Iterate over rows, buckets, and sub-buckets to decay each count value
        for i in range(self.num_rows):
            for j in range(self.num_buckets):
                for k in range(self.num_buckets):
                    
                    # Apply decay to the count value
                    self.count[i][j][k] = self.count[i][j][k] * decay_factor

    def getAnoEdgeGlobalScore(self, src, dst):
        """
        Computes the anomaly score for an edge (src, dst) based on local density.

        Parameters
        ----------
        src : int
            Source node of the edge.
        dst : int
            Destination node of the edge.

        Returns
        -------
        min_dsubgraph : float
            Anomaly score for the edge.

        """
        
        # Initialize minimum dense subgraph score to positive infinity
        min_dsubgraph = float('inf')
        
        # Iterate over the rows to calculate the density of each submatrix
        for i in range(self.num_rows):
            
            # Determine the buckets using the hash function
            src_bucket = self.hash(src, i)
            dst_bucket = self.hash(dst, i)
            
            # Calculate the density of the current submatrix
            cur_dsubgraph = AnoEdgeGlobal.getAnoEdgeGlobalDensity(self.count[i], src_bucket, dst_bucket)
            
            # Update the minimum dense subgraph score
            min_dsubgraph = min(min_dsubgraph, cur_dsubgraph)
            
        return min_dsubgraph

    def getAnographScore(self):
        """
        Computes the global anomaly score for the entire count matrix based on local densities.

        Returns
        -------
        min_dsubgraph : float
            Global anomaly score for the entire count matrix.

        """
        
        # Initialize minimum dense subgraph score to positive infinity
        min_dsubgraph = float('inf')
        
        # Iterate over the rows to calculate the density of each submatrix
        for i in range(self.num_rows):
            
            # Calculate the density of the current submatrix
            cur_dsubgraph = Anograph.getAnographDensity(self.count[i])
            
            # Update the minimum dense subgraph score
            min_dsubgraph = min(min_dsubgraph, cur_dsubgraph)
            
        return min_dsubgraph


class Anograph:
    """
    Anograph class for computing anomaly scores based on matrix density calculations.
    
    Methods
    -------
    learn_one(self, x) :
        Update the Anograph instance with a new edge.
        
    get_score(self) :
        Get the current anomaly score.
        
    pickMinRow(mat, row_flag, col_flag) : static method
        Pick the row with the minimum sum from the matrix.
        
    pickMinCol(mat, row_flag, col_flag) : static method
        Pick the column with the minimum sum from the matrix.
        
    getMatrixDensity(mat, row_flag, col_flag) : static method
        Compute the density of the submatrix specified by row_flag and col_flag.
    
    getAnographDensity(mat) :
        Compute the Anograph density of the matrix
    """
    
    def __init__(self, time_window, edge_threshold, rows, buckets):
        """
        Parameters
        ----------
        time_window : int
            Time window for processing batches of edges.
        edge_threshold : int
            Threshold for edge weights in the anomaly detection process.
        rows : int
            Number of rows in the sketch matrix for density calculations.
        buckets : int
            Number of buckets in each row of the sketch matrix.

        Returns
        -------
        None.

        """
        
        self.time_window = time_window
        self.edge_threshold = edge_threshold
        self.rows = rows
        self.buckets = buckets
        self.graph = [] # List to store the processed edges
        self.score = 0.0 # Current anomaly score
        self.cur_time = 0 # Current time window
        self.cur_src = [] # List to store source nodes of the current batch
        self.cur_dst = [] # List to store destination nodes of the current batch
        self.cur_count = Hcms(self.rows, self.buckets) # Instance of the Hcms class for counting matrix calculations


    def learn_one(self, x):
        """
        Update the Anograph instance with a new edge.

        Parameters
        ----------
        x : dict
            Dictionary containing edge information (source, destination, timestamp).

        Returns
        -------
        Updated Anograph instance

        """
        
        # Extract source, destination, and timestamp from the input dictionary
        s, d, t = map(int, x.values())
        
        # Check if the timestamp falls into a new time window
        if t // self.time_window != self.cur_time:
            
            #Update HCMS with the current batch of edges
            src, dst = (self.cur_src, self.cur_dst)
            num_edges = len(src)
            for j in range(num_edges):
                self.cur_count.insert(src[j], dst[j], 1) #1 stands for the weight of the edge
            self.cur_time = t // self.time_window
            self.score = self.cur_count.getAnographScore()
            
            #Empty batch
            self.cur_src, self.cur_dst = [], []
            
            #Initialize a new HCMS for the next time window
            self.cur_count = Hcms(self.rows, self.buckets)

        self.cur_src.append(s)
        self.cur_dst.append(d)

        return self
    
    def get_score(self):
        """
        Get the current anomaly score.
        

        Returns
        -------
        float
            Current anomaly score.

        """
        
        return self.score


    @staticmethod
    def pickMinRow(mat, row_flag, col_flag):
        """
        Pick the row with the minimum sum from the matrix.

        Parameters
        ----------
        mat : list
            2D matrix.
        row_flag : list
            List of flags indicating whether a row is selected.
        col_flag : list
            List of flags indicating whether a column is selected.

        Returns
        -------
        ans : tuple
            Index and sum of the selected row.

        """
        
        num_rows = len(mat)
        num_cols = len(mat[0])
        ans = (-1, float('inf'))
        
        # Iterate over rows to find the one with the minimum sum
        for i in range(num_rows):
            if row_flag[i]:
                
                # Calculate the sum of selected columns in the current row
                row_sum = sum(mat[i][j] for j in range(num_cols) if col_flag[j])
                
                # Update the minimum sum and index if the current row has a smaller sum
                if row_sum < ans[1]:
                    ans = (i, row_sum)
        return ans

    @staticmethod
    def pickMinCol(mat, row_flag, col_flag):
        """
        Pick the column with the minimum sum from the matrix.

        Parameters
        ----------
        mat : list
            2D matrix.
        row_flag : list
            List of flags indicating whether a row is selected.
        col_flag : list
            List of flags indicating whether a column is selected.

        Returns
        -------
        ans : tuple
            Index and sum of the selected row.
            
        """
        num_rows = len(mat)
        num_cols = len(mat[0])
        ans = (-1, float('inf'))
        
        # Iterate over columns to find the one with the minimum sum
        for i in range(num_cols):
            if col_flag[i]:
                
                # Calculate the sum of selected rows in the current column
                col_sum = sum(mat[j][i] for j in range(num_rows) if row_flag[j])
                
                # Update the minimum sum and index if the current column has a smaller sum
                if col_sum < ans[1]:
                    ans = (i, col_sum)
        return ans


    @staticmethod
    def getMatrixDensity(mat, row_flag, col_flag):
        """
        Compute the density of the submatrix specified by row_flag and col_flag.

        Parameters
        ----------
        mat : list
            2D matrix.
        row_flag : list
            List of flags indicating whether a row is selected.
        col_flag : list
            List of flags indicating whether a column is selected.

        Returns
        -------
        float
            Density of the submatrix.

        """
        
        num_rows = len(mat)
        num_cols = len(mat[0])
        row_ctr = 0.0
        col_ctr = 0.0
        ans = 0.0
        
        # Iterate over rows and columns to calculate the density of the selected submatrix
        for i in range(num_rows):
            if row_flag[i]:
                row_ctr += 1.0
                for j in range(num_cols):
                    if col_flag[j]:
                        ans += mat[i][j]
                        
        # Count the number of selected columns
        for j in range(num_cols):
            if col_flag[j]:
                col_ctr += 1.0
        
        # Avoid division by zero and return the calculated density
        if row_ctr == 0.0 or col_ctr == 0.0:
            return 0.0
        return ans / ((row_ctr * col_ctr) ** 0.5)
    
    @staticmethod
    def getAnographDensity(mat):
        """
        Compute the Anograph density of the matrix

        Parameters
        ----------
        mat : list
            2D matrix.

        Returns
        -------
        output : float
            Anograph density of the matrix.

        """
        
        num_rows = len(mat)
        num_cols = len(mat[0])
 
        # Initialize flags for all rows and columns to be selected
        row_flag = [True] * num_rows
        col_flag = [True] * num_cols

        # Initialize output with the density of the entire matrix
        output = Anograph.getMatrixDensity(mat, row_flag, col_flag)

        # Iterate to pick and exclude rows or columns to maximize density
        ctr = num_rows + num_cols
        while ctr > 0:
            
            # Pick the row and column with the minimum sum
            picked_row = Anograph.pickMinRow(mat, row_flag, col_flag)
            picked_col = Anograph.pickMinCol(mat, row_flag, col_flag)

            if picked_row[1] <= picked_col[1]:
                
                # Exclude the picked row and recalculate density
                row_flag[picked_row[0]] = False
                mat_sum = Anograph.getMatrixDensity(mat, row_flag, col_flag)

            else:
                
                # Exclude the picked column and recalculate density
                col_flag[picked_col[0]] = False
                mat_sum = Anograph.getMatrixDensity(mat, row_flag, col_flag)
                
            # Update output with the maximum density obtained
            output = max(output, mat_sum)
            ctr -= 1

        return output
    


class AnoEdgeGlobal:
    """
    Anomaly Detection using Edge Global Density.
    
    Methods
    -------
    learn_one(self, x) : 
        Update the AnoEdgeGlobal instance with a new edge.
        
    score_one(self, x) :
        Calculate the anomaly score for a given edge.
        
    getAnoEdgeGlobalDensity(mat, src, dst) : static method
        Calculate the density-based anomaly score for a specific edge.
        
    """
    
    def __init__(self, rows, buckets, decay_factor):
        """
        Parameters
        ----------
        rows : int
            Number of rows in the Hcms count matrix.
        buckets : int
            Number of buckets in the Hcms count matrix.
        decay_factor : float
            Decay factor applied to the Hcms count matrix.

        Returns
        -------
        None.

        """
        
        self.rows = rows
        self.buckets = buckets
        self.decay_factor = decay_factor
        self.last_time = 0
        
        # Initialize the count matrix with the specified number of rows and buckets
        self.count = Hcms(self.rows, self.buckets)


    def learn_one(self, x):
        """
        Update the AnoEdgeGlobal instance with a new edge.

        Parameters
        ----------
        x : dict
            Dictionary containing edge information (source, destination, timestamp).

        Returns
        -------
        Updated AnoEdgeGlobal instance.

        """
        
        # Extract source, destination, and timestamp from the input dictionary
        src, dst, time = map(int, x.values())
        
        # Check if the timestamp is greater than the last recorded time
        if time > self.last_time:
            
            # If yes, decay the count matrix using the specified decay factor
            self.count.decay(self.decay_factor)
        
        # Insert the new edge into the count matrix with a weight of 1
        self.count.insert(src, dst, 1)
        
        # Update the last recorded time
        self.last_time = time
        
        # Return the updated AnoEdgeGlobal instance
        return self
    
    def score_one(self, x):
        """
        Calculate the anomaly score for a given edge.

        Parameters
        ----------
        x : dict
            Dictionary containing edge information (source, destination, timestamp).

        Returns
        -------
        score : float
            Anomaly score for the given edge.

        """
        
        # Extract source, destination, and timestamp from the input dictionary
        src, dst, _ = map(int, x.values())
        
        # Calculate the anomaly score for the specified edge using the count matrix
        score = self.count.getAnoEdgeGlobalScore(src, dst)
        
        # Return the calculated anomaly score
        return score


    @staticmethod
    def getAnoEdgeGlobalDensity(mat, src, dst):
        """
        Calculate the density-based anomaly score for a specific edge.

        Parameters
        ----------
        mat : list
            2D matrix.
        src : int
            Source node index.
        dst : int
            Destination node index.

        Returns
        -------
        output : float
            Density of the specified submatrix.

        """
        
        num_rows = len(mat)
        num_cols = len(mat[0])

        # Initialize flags and slice sums for rows and columns
        row_flag = [False] * num_rows
        col_flag = [False] * num_cols

        row_slice_sum = [mat[i][dst] for i in range(num_rows)]
        col_slice_sum = [mat[src][i] for i in range(num_cols)]

        # Mark the source and destination nodes
        row_flag[src] = True
        col_flag[dst] = True
        row_slice_sum[src] = mat[src][dst]
        col_slice_sum[dst] = mat[src][dst]

        # Find the initial maximum row and column
        max_row = (-1, -1.0)
        for i in range(num_rows):
            if not row_flag[i] and row_slice_sum[i] >= max_row[1]:
                max_row = (i, row_slice_sum[i])

        max_col = (-1, -1.0)
        for i in range(num_cols):
            if not col_flag[i] and col_slice_sum[i] >= max_col[1]:
                max_col = (i, col_slice_sum[i])

        # Initialize counters for marked rows and columns, and the current matrix sum
        marked_rows = 1
        marked_cols = 1

        cur_mat_sum = mat[src][dst]
        
        # Initialize the output with the initial density
        output = cur_mat_sum / ((marked_rows * marked_cols) ** 0.5)
        
        # Initialize the counter for iterations
        ctr = num_rows + num_cols - 2
        
        # Iterate to find the maximum density
        while ctr > 0:
            if max_row[1] >= max_col[1]:
                row_flag[max_row[0]] = True
                marked_rows += 1
                
                # Update maximum column and current matrix sum for marked row
                max_col = (-1, -1.0)
                for i in range(num_cols):
                    if col_flag[i]:
                        cur_mat_sum += mat[max_row[0]][i]
                    else:
                        col_slice_sum[i] += mat[max_row[0]][i]
                        if col_slice_sum[i] >= max_col[1]:
                            max_col = (i, col_slice_sum[i])
                            
                # Find the new maximum row
                max_row = (-1, -1.0)
                for i in range(num_rows):
                    if not row_flag[i] and row_slice_sum[i] >= max_row[1]:
                        max_row = (i, row_slice_sum[i])

            else:
                col_flag[max_col[0]] = True
                marked_cols += 1
                
                # Update maximum row and current matrix sum for marked column
                max_row = (-1, -1.0)
                for i in range(num_rows):
                    if row_flag[i]:
                        cur_mat_sum += mat[i][max_col[0]]
                    else:
                        row_slice_sum[i] += mat[i][max_col[0]]
                        if row_slice_sum[i] >= max_row[1]:
                            max_row = (i, row_slice_sum[i])

                # Find the new maximum column
                max_col = (-1, -1.0)
                for i in range(num_cols):
                    if not col_flag[i] and col_slice_sum[i] >= max_col[1]:
                        max_col = (i, col_slice_sum[i])
            
            # Update output with the maximum density obtained
            output = max(output, cur_mat_sum / ((marked_rows * marked_cols) ** 0.5))
            ctr -= 1

        return output




