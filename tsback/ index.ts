type ParseInt<T extends string> = T extends `${infer Digit extends number}`
  ? Digit
  : never;

type ReverseString<S extends string> = S extends `${infer Frist}${infer Rest}`
  ? `${ReverseString<Rest>}${Frist}`
  : "";

type RemoveLeadingZeros<S extends string> = S extends "0"
  ? S
  : S extends `0${infer R}`
  ? RemoveLeadingZeros<R>
  : S;

type PutSign<S extends string> = `-${S}`;

// `S` is a reversed string representing some number, e.g., "0321" instead of 1230
type InternalMinusOne<S extends string> =
  S extends `${infer Digit extends number}${infer Rest}`
    ? Digit extends 0
      ? `9${InternalMinusOne<Rest>}`
      : `${[9, 0, 1, 2, 3, 4, 5, 6, 7, 8][Digit]}${Rest}`
    : never;

type AlmostFullMinusOne<T extends number> = ParseInt<
  RemoveLeadingZeros<ReverseString<InternalMinusOne<ReverseString<`${T}`>>>>
>;

type InternalPlusOne<S extends string> = S extends "9"
  ? "01"
  : S extends `${infer Digit extends number}${infer Rest}`
  ? Digit extends 9
    ? `0${InternalPlusOne<Rest>}`
    : `${[1, 2, 3, 4, 5, 6, 7, 8, 9][Digit]}${Rest}`
  : never;

export type MinusOne<T extends number> = T extends 0
  ? -1 // T = 0
  : `${T}` extends `-${infer Abs}`
  ? ParseInt<PutSign<ReverseString<InternalPlusOne<ReverseString<Abs>>>>> // T < 0
  : // T > 0
    ParseInt<
      RemoveLeadingZeros<ReverseString<InternalMinusOne<ReverseString<`${T}`>>>>
    >;

// type MinusOne<n extends number>  = n extends 0?
type Repeat<
  V extends string,
  count extends number,
  acc extends string = ""
> = count extends 0 ? acc : Repeat<V, MinusOne<count>, `${acc}${V}`>;

type Create10ToPower<Power extends number> = ParseInt<`1${Repeat<"0", Power>}`>;
