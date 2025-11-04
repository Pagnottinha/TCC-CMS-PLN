"""Stub para helpers.storage"""

class BitVector:
    """Implementation of a bit vector.
    
    Attributes
    ----------
    length : int
        The length of the vector's index space.
    size : int
        The size of the array of BitFields.
    """
    
    length: int
    size: int
    
    def __init__(self, length: int) -> None:
        """Allocate and initialize a bit vector.
        
        Parameters
        ----------
        length : int
            The length of the vector.
        """
        ...
    
    def __getitem__(self, index: int) -> bool:
        """Get element (bit value) by the index.
        
        Parameters
        ----------
        index : int
            The index of the element in the vector.
        
        Returns
        -------
        bool
            True if bit is set, False otherwise.
        
        Raises
        ------
        IndexError
            If `index` is out of range.
        """
        ...
    
    def __setitem__(self, index: int, flag: bool) -> None:
        """Set element (bit value) by the index.
        
        Parameters
        ----------
        index : int
            The index of the element in the vector.
        flag : bool
            Value to set (True or False).
        
        Raises
        ------
        IndexError
            If `index` is out of range.
        """
        ...
    
    def __len__(self) -> int:
        """Get length of the vector's index space.
        
        Returns
        -------
        int
            The length of the vector's index space.
        """
        ...
    
    def __repr__(self) -> str:
        """String representation of the BitVector.
        
        Returns
        -------
        str
            String representation.
        """
        ...
    
    def sizeof(self) -> int:
        """Size of the vector in bytes.
        
        Returns
        -------
        int
            Number of bytes allocated for the vector.
        """
        ...
    
    def count(self) -> int:
        """Count number of set bits in the vector.
        
        Returns
        -------
        int
            Number of set bits in the vector.
        """
        ...

